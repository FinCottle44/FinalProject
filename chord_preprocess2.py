# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import sys
import io
import zipfile
import miditoolkit
import random
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager, Value

# Music21 Import:
import sys
sys.path.append('/root/project/venv/lib/python3.7/site-packages')
from music21 import *
import re

# ChordTuple Params
c_trans = False
chord_reduce = False

# MusicBERT original params
pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 2000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127

data_zip = None
output_file = None


lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
manager = Manager()
midi_dict = manager.dict()

###### MusicBERT Code Section:
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)

def writer(file_name, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            f.write(output_str + '\n')
            
###### End of MusicBERT Code Section

## Custom code - Based on MusicBERT preprocess.gen_dictionary()
def gen_chord_dictionary(file_name):
    num = 0
    with open(file_name, 'w') as f:
        # Chord information
        for j in range(bar_max): # Bar
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution): # Pos
            print('<1-{}>'.format(j), num, file=f)
        for j in range(-1, 2 * max_pitch + 1 + 1): # Pitches (-1 for pad token)
            print('<2-{}>'.format(j), num, file=f)                                          

###### Custom Code Fin Cottle 1938561
def get_chords(m21_data, eval=False, chord_filt=False):
    # Separate instruments
    instrument_streams = instrument.partitionByInstrument(m21_data)
    if instrument_streams is None:
        return []

    # Find most suited instrument
    # - Looking for over 4 chords with atleast 3 notes
    best_progs = []
    for inst in instrument_streams:
        # Ignore percussive instruments as can cause issues with chordify()
        if isinstance(inst, instrument.Percussion):
            continue
        
        # Fetch elements
        try:
            # chords = inst.chordify().recurse().getElementsByClass(chord.Chord)
            chords = inst.recurse().getElementsByClass(chord.Chord)
        except AttributeError as e:
            # Sometimes chords in a MIDI file are interpreted as PercussionChords by Music21,
            # these have no 'pitch' attribute, so chordify() breaks...
            print(f"Tried processing percussion chord on instrument {inst}")
            continue
        
        # More checks
        if chord_filt:
            allowed_qualities = ['major', 'minor'] # allows most chords, just filters horrible ones
        
        if len(chords) == 0: # If no chords, next iteration
            continue

        prog = []
        num_pitches = 0
        unique_chords = set()
        for c in chords:
            c.pitches = set(c.pitches)
            if tuple(c.pitchNames) in unique_chords:
                continue
            
            unique_chords.add(tuple(c.pitchNames)) # pitchnames is changed when changing c.pitches
            num_pitches += len(c.pitches)

            ## Apply filters:
            if len(c.pitches) < 3: # want atleast triads
                continue

            if c.duration.quarterLength <= 0.2: # Remove slight 'grace' chords
                continue
            
            # # Reduce to single octave:
            if chord_reduce:
                c.pitches = reduce_to_single_octave(c)

            # Passed filters
            prog.append(c)
        
        # Calculate AP (avg number of notes in chords)
        average_polyphony = num_pitches / len(unique_chords) if len(unique_chords) > 0 else 0
        
        if eval: # If we're using this function to evaluate (ie Test) then return the chords in test file
            return chords
        else: # Else, we are clearly pre-processing lots of data and trying to get valuable progressions so:
            if len(unique_chords) >= 4 and average_polyphony > 2.5: # almost all triads
                print(f"Track {inst.getInstrument()} GOOD quality: {len(unique_chords)} chords averaging {average_polyphony} notes")
                best_progs.append(prog)
            else:
                print(f"Track {inst.getInstrument()} BAD quality: {len(unique_chords)} chords averaging {average_polyphony} notes")
    return best_progs

## Custom Code Fin Cottle 1938561
def transpose_to_c(chords):
    # Create part for adding chords
    s = stream.Stream()
    
    # Add chords
    for c in chords:
        s.append(c)
    
    # Analyze key
    key = s.analyze('key')
    if key.mode == "minor":
        key = key.relative # Ensure in Major
    
    # Transpose
    i = interval.Interval(key.tonic, pitch.Pitch('C'))
    s.transpose(i, inPlace=True)
    chords = s.getElementsByClass(chord.Chord)    
    difference = int(i.cents/100)
    
    print(f"Transposed original chord list  in {key} by {difference} to C Maj)")
    return chords, difference

####### Custom Code Fin Cottle 1938561
def reduce_to_single_octave(c):
    c = [p.nameWithOctave for p in c.pitches] # Format chord as list from numbers: [60, 64, 67] to names: ['C5', 'E5', 'G5']
    notes = {} # Dict for a pitch ('C') and respective octaves (['C4', 'C5'])
    
    # Iterate notes in chord
    for note in c:
        note_no_octave = note[0] # strip octave
        if note_no_octave not in notes:
            notes[note_no_octave] = [] # Create entry in dict if not already there
        notes[note_no_octave].append(note)
    
    # Create a new chord and add notes in a single octage
    reduced_chord = []
    for note, octave_notes in notes.items():
        if len(octave_notes) > 1: # If more than 1 note, sort, find highest and keep
            highest_octave_note = max(octave_notes, key=lambda x: int(x[-1]))
            reduced_chord.append(highest_octave_note)
        else:
            reduced_chord.append(octave_notes[0]) # Keep original as just 1
    return reduced_chord

def chord_list_to_encoding(chords):
    encoding = []
    count = 0 # Value for chord position
    for c in chords:
        c = c.sortAscending() # Chords should always be consistent for pattern matching

        # Pad unused pitches with -1
        pitches = [p.midi for p in c.pitches] + [-1] * (6 - len(c.pitches))

        #old
        # encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                         # 1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))

        # new: (0 Count, 1 Pos, 2 Pitch 1, 3 Pitch 2, 4 Pitch 3, 5 Pitch 4, 6 Pitch 5, 7 Pitch 6)
        encoding.append((count, 0, 
                         pitches[0], pitches[1], 
                         pitches[2], pitches[3], 
                         pitches[4], pitches[5]
                        ))
        count += 1 # Increments each iteration
    return encoding


def chord_encoding_to_MIDI(encoding):
    ## Refined MusicBERT preprocess.encoding_to_MIDI
    # Initialise MIDI file
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    
    # Iterate encoding and create respective midi notes
    for i in encoding:
        # new: (0 Measure, 1 Pos, 2 Pitch 1, 3 Pitch 2, 4 Pitch 3, 5 Pitch 4, 6 Pitch 5, 7 Pitch 6)
        start = (midi_obj.ticks_per_beat * 4 * i[0]) + i[1] # (ticks_per_beat * 4 (time sig) * bar) + position
        pitches = i[2:8]
        end = start + (midi_obj.ticks_per_beat * 4) # each ch
        for p in pitches:
            midi_obj.instruments[0].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=p, velocity=100))
    
    # Revert to necessary instruments
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    return midi_obj


# MusicBERT Code:
def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash

# Adapted MusicBERT preprocess.F() function
def F_chord(file_name):
    # Check if can parse
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    
    # Check if blank
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    
    try:            
        m21_obj = converter.parse(file_name)

        ## Get most valuable chord progression from midi
        best_progressions = get_chords(m21_obj, chord_filt=False)
        if len(best_progressions) < 1:
            print('ERROR(QUALITY): ' + file_name + '\n', end='') #Midi had no good chord progressions or could not be split by instrument
            return None
        
        for chords in best_progressions: 
            ## Transpose to c
            if c_trans:
                chords, difference = transpose_to_c(chords)

            ## Encode
            e = chord_list_to_encoding(chords)

            if len(e) == 0:
                print('ERROR(BLANK): ' + file_name + '\n', end='')
                return None
            
            output_str_list = []
            sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
            for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
                L = max(p, 0)
                R = min(p + sample_len_max, len(e)) - 1
                bar_index_list = [e[i][0]
                                    for i in range(L, R + 1) if e[i][0] is not None]
                bar_index_min = 0
                bar_index_max = 0
                if len(bar_index_list) > 0:
                    bar_index_min = min(bar_index_list)
                    bar_index_max = max(bar_index_list)
                offset_lower_bound = -bar_index_min
                offset_upper_bound = bar_max - 1 - bar_index_max
                # to make bar index distribute in [0, bar_max)
                bar_index_offset = random.randint(
                    offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
                e_segment = []
                for i in e[L: R + 1]:
                    if i[0] is None or i[0] + bar_index_offset < bar_max:
                        e_segment.append(i)
                    else:
                        break
                tokens_per_chord = 8
                
                # Actual encoding
                output_words = (['<s>'] * tokens_per_chord) \
                    + [('<{}-{}>'.format(min(j,2), k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
                    + (['</s>'] * (tokens_per_chord - 1)
                        )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
                output_str_list.append(' '.join(output_words))

        # no empty toks
        if not all(len(i.split()) > tokens_per_chord * 2 - 1 for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        try:
            lock_write.acquire()
            writer(file_name, output_str_list)
        except BaseException as e:
            print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        finally:
            lock_write.release()
    finally:
        # Increment counter 
        global counter
        with counter.get_lock():
            counter.value += 1
        if counter.value % 100 == 0:
            print("Saving example chord file at midi/processed_example.mid")

        print(f"Processed {counter.value} out of {total_file_cnt} (~{round(counter.value / total_file_cnt * 100, 2) }%)")
        
    print('SUCCESS: ' + file_name + '\n', end='')
    return True

# Slightly Adapted MusicBERT preprocess.G function
# If there are any unexpected exceptions, prevents breaking of execution of all the data.
def G_chord(file_name):
    try:
        return F_chord(file_name) # Expected exceptions are handled here
    except BaseException as e:
        print(e)
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False


def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    # TODO - Remove tokens per note and try use with MusicBERT
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
    return encoding


#### Edits - Fin Cottle c1938561
# Based on MusicBERT encoding_to_str()
def chord_encoding_to_str(e):
    # Based on MusicBERT encoding_to_str()
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(min(j,2), k if j > 0 else k + bar_index_offset) for i in e[p: p +
                                                                                            sample_len_max] if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]
                    + (['</s>'] * (tokens_per_note
                                   - 1)))   # 8 - 1 for append_eos functionality of binarizer in fairseq

def get_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        print(f"Searching: {root}")
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

#### Edits - Fin Cottle c1938561
# Counter so can track progress of preprocessing
counter = None
def init(args):
    print("Initialising counter")
    global counter
    counter = args

if __name__ == "__main__":
    # Begin:
    data_path = f"{input('Dataset path: ')}"
    print(f"Set data path: {data_path}")
    prefix = f"{input('OctupleMIDI output path: ')}_data_raw"
    if os.path.exists(prefix):
        print('Output path {} already exists!'.format(prefix))
        sys.exit(0)
    print(f"Set output: {prefix}")
    os.system('mkdir -p {}'.format(prefix))
    
    print("Getting file paths...")
    file_list = get_file_paths(data_path)
    print("Shuffling...")
    random.shuffle(file_list)
    print("Generating dict...")
    gen_chord_dictionary('{}/dict.txt'.format(prefix))
    ok_cnt = 0
    all_cnt = 0
    for sp in ['train', 'valid', 'test']:
        global total_file_cnt
        total_file_cnt = len(file_list)   
        print(f"Total files: {total_file_cnt}")  
        file_list_split = []
        if sp == 'train':  # 98%
            file_list_split = file_list[: 98 * total_file_cnt // 100]
        if sp == 'valid':  # 1%
            file_list_split = file_list[98 * total_file_cnt //
                                        100: 99 * total_file_cnt // 100]
        if sp == 'test':  # 1%
            file_list_split = file_list[99 * total_file_cnt // 100:]
        output_file = '{}/midi_{}.txt'.format(prefix, sp)

        print(f"File split: {sp}")
        # Counter obj
        counter = Value('i', 0)
            
        with Pool(pool_num, initializer=init, initargs=(counter, )) as p:
            result = list(p.imap_unordered(G_chord, file_list_split))
            all_cnt += sum((1 if i is not None else 0 for i in result))
            ok_cnt += sum((1 if i is True else 0 for i in result))
        output_file = None
    print('{}/{} ({:.2f}%) MIDI files successfully processed'.format(ok_cnt, all_cnt, ok_cnt / all_cnt * 100))