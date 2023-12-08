# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import os
import sys
import miditoolkit
import random
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager, Value

# Custom - Music21 Import:
import sys
sys.path.append('/root/project/venv/lib/python3.7/site-packages')
from music21 import *

## MusicBERT Code:
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
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

data_zip = None
output_file = None


lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
manager = Manager()
midi_dict = manager.dict()


# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]


###### MusicBERT Code Section:
ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)


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


def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]


def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator

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
        for j in range(bar_max):
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('<1-{}>'.format(j), num, file=f)
        for j in range(duration_max * pos_resolution):
            print('<2-{}>'.format(j), num, file=f)
        for j in range(-1, 2 * max_pitch + 1 + 1): # -1 as need padding token
            print('<3-{}>'.format(j), num, file=f)

## Custom Code Fin Cottle 1938561
def get_chords(m21_data, ticks_per_beat, eval=False):
    debug = False 
    # Separate instruments
    instrument_streams = instrument.partitionByInstrument(m21_data)
    if instrument_streams is None:
        return None

    # Find most suited instrument
    # - Looking for over 4 chords with atleast 3 notes
    best_prog = []
    for inst in instrument_streams:
        chords = list(inst.getElementsByClass(chord.Chord))
        allowed_qualities = ['major', 'minor'] # allows most chords, just filters horrible ones
        good_prog = []
        
        if len(chords) == 0: # If no chords
            continue

        excl_count = 0
        for c in chords:
            # Remove duplicate notes (e.g. E4 and E4)
            c.pitches = set(c.pitches)
            
            if not eval: # Preprocessing so want good data
                ## Apply filters:
                if len(c.pitches) < 3: # want atleast triads
                    excl_count += 1
                    continue
                
                if c.quality not in allowed_qualities: # Want only majors and minors
                    excl_count += 1
                    continue

                if c.duration.quarterLength <= 0.5: # Remove slight 'grace' chords
                    excl_count += 1
                    continue
                
                # Reduce to single octave:
                c.pitches = reduce_to_single_octave(c)

            # Passed filters
            good_prog.append(c) 
        
        if not eval:
            print(f"Excluded {excl_count} of {len(chords)} chords.")
        

        if len(good_prog) > len(best_prog): # Get best progression in file
            best_prog = good_prog
            if debug:
                print(f"------\nFound instrument {inst} with best chord progression:")
                print(f"{best_prog}")
    
    if len(best_prog) < 3:
        return None
    
    prev_chord_end = 0
    for c in best_prog:
        c.midiTickStart = prev_chord_end
        midi_dur = c.duration.quarterLength * ticks_per_beat
        prev_chord_end = prev_chord_end + midi_dur
    
    return best_prog

## Custom Code Fin Cottle 1938561
def transpose_to_c(chords):
    # Create part for adding chords
    s = stream.Stream()
    
    # Add chords
    for c in chords:
        s.append(c)
    
    # Analyze key
    key = s.analyze('key')
    if key.mode == "minor" :
        # print(f"Detected minor ({key}) so found relative major ({key.relative})")
        key = key.relative
    
    # Transpose
    i = interval.Interval(key.tonic, pitch.Pitch('C'))
    s.transpose(i, inPlace=True)
    chords = s.getElementsByClass(chord.Chord)
    # midi = midi.transpose(i) # Iterate chords instead
    
    difference = int(i.cents/100)
    print(f"Transposed original chord list  in {key} by {difference} to C Maj)")
    return chords, difference

####### Custom Code Fin Cottle 1938561
def reduce_to_single_octave(c):
    c = [p.nameWithOctave for p in c.pitches] # Format chord as list from numbers: [60, 64, 67] to names: ['C5', 'E5', 'G5']
    notes = {} # Dict for a pitch ('C') and respective octaves (['C4', 'C5'])
    
    # Iterate notes in chord
    for note in c:
        pitch_class = note[0] # strip octave
        if pitch_class not in notes:
            notes[pitch_class] = [] # Create entry in dict if not already there
        notes[pitch_class].append(note)
    
    # Create a new chord and add notes in a single octage
    reduced_chord = []
    for note, octave_notes in notes.items():
        if len(octave_notes) > 1: # If more than 1 note, sort, find highest and keep
            highest_octave_note = max(octave_notes, key=lambda x: int(x[-1])) # Sort by last char in pitch string 'C5' -> 5
            reduced_chord.append(highest_octave_note)
        else:
            reduced_chord.append(octave_notes[0]) # Keep original as just 1
    return reduced_chord

def chord_list_to_encoding(chords, midi_obj):
    ## MusicBERT code:
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    
    ## Original MusicBERT code:
    # notes_start_pos = [time_to_pos(j.start)
    #                    for i in midi_obj.instruments for j in i.notes]
    ## Changed code:
    notes_start_pos = [time_to_pos(c.midiTickStart)
                       for c in chords]
    
    ## MusicBERT positional code
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)

    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution

    # Custom Code Fin Cottle 1938561:
    for c in chords:
        c = c.sortAscending().removeRedundantPitches() # Sort ascending for consistency
        if time_to_pos(c.midiTickStart) >= trunc_pos: # Continue if too long
            continue

        # Pad pitches with -1
        pitches = [p.midi for p in c.pitches] + [-1] * (5 - len(c.pitches))

        # Get timing & start information from MusicBERT functions
        start_distribution[time_to_pos(c.midiTickStart) % pos_resolution] += 1
        info = pos_to_info[time_to_pos(c.midiTickStart)]

        ## MusicBERT Encoding: 
        # encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                         # 1 if inst.is_drum 
                         # else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), 
                         #                                  v2e(note.velocity), info[1], info[3]))

        ## Changed code:
        # new: (0 Measure, 1 Pos, 2 Duration, 3 Pitch 1, 4 Pitch 2, 5 Pitch 3, 6 Pitch 4, 7 Pitch 5)
        encoding.append((info[0], info[2], 
                         d2e(int(midi.translate.durationToMidiTicks(c.duration) / pos_resolution / 4)),
                         # '/ 4' as always dealing with 4/4 time sig in ChordTuple                          
                         pitches[0], pitches[1], 
                         pitches[2], pitches[3], 
                         pitches[4]
                        ))
    
    # If empty return empty list
    if len(encoding) == 0:
        return list()

    encoding.sort()
    return encoding


def chord_encoding_to_MIDI(encoding):
    ## Mostly MusicBERT preprocess.encoding_to_MIDI
    bar_to_pos = [None] * len(range(max(map(lambda x: x[0], encoding)) + 1)) # List of 'None' for all bars (iterates encoding to find start of last chord)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        # ts = e2t(bar_to_timesig[i])
        ts = (4,4)
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length 
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        #### EDITS: Fin Cottle 1938561
        # new: (0 Measure, 1 Pos, 2 Duration, 3 Pitch 1, 4 Pitch 2, 5 Pitch 3, 6 Pitch 4, 7 Pitch 5)
        start = get_tick(i[0], i[1])
        # duration = e2d(i[2])
        duration = get_tick(0, e2d(i[2]))
        # pitch = (i[3] - 128 if program == 128 else i[3])
        pitches = i[3:8]
        if duration == 0:
            duration = 1
        end = start + duration
        # velocity = e2v(i[5])
        for p in pitches:
            midi_obj.instruments[0].notes.append(miditoolkit.containers.Note(
                start=start, end=end, pitch=p, velocity=100))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    return midi_obj
####### End Custom Code Section

# MusicBERT Code:
def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash

# Adapted MusicBERT preprocess.F() function
def F_chord(file_name):
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
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    
    try:            
        m21_obj = converter.parse(file_name)
        mtk_obj = miditoolkit.midi.parser.MidiFile(file_name)

        ## Get most valuable chord progression from midi
        chords = get_chords(m21_obj, mtk_obj.ticks_per_beat)
        if chords is None:
            print('ERROR(QUALITY): ' + file_name + '\n', end='') #Midi had no good chord progressions or could not be split by instrument
            return None

        ## Encode
        e = chord_list_to_encoding(chords, mtk_obj)

        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            lock_set.release()
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                        file_name + ' == ' + dup_file_name + '\n', end='')
                return None
            
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
            tokens_per_note = 8
            output_words = (['<s>'] * tokens_per_note) \
                + [('<{}-{}>'.format(min(j,3), k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
                + (['</s>'] * (tokens_per_note - 1)
                    )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
            output_str_list.append(' '.join(output_words))

        # no empty
        if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
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

        print(f"Processed {counter.value} out of {total_file_cnt} (~{counter.value // total_file_cnt * 100}%)")
        
    print('SUCCESS: ' + file_name + '\n', end='')
    return True

# Slightly Adapted MusicBERT preprocess.G function
def G_chord(file_name):
    try:
        return F_chord(file_name)
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False


def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
    return encoding


#### Edits - Fin Cottle c1938561
# Based on MusicBERT encoding_to_str()
def chord_encoding_to_str(e):
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(min(j,3), k if j > 0 else k + bar_index_offset) for i in e[p: p +
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
    prefix = f"{input('OctupleMIDI output path: ')}"
    print(f"Set output: {prefix}")
    if os.path.exists(prefix):
        print('Output path {} already exists!'.format(prefix))
        sys.exit(0)
    os.system('mkdir -p {}'.format(prefix))
    # data_zip = zipfile.ZipFile(data_path, 'r')
    # file_list = [n for n in data_path.namelist() if n[-4:].lower() == '.mid' or n[-5:].lower() == '.midi']
    
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
        
        # Original MusicBERT pre-processign
        with Pool(pool_num, initializer=init, initargs=(counter, )) as p:
            result = list(p.imap_unordered(G_chord, file_list_split))
            all_cnt += sum((1 if i is not None else 0 for i in result))
            ok_cnt += sum((1 if i is True else 0 for i in result))
        output_file = None
    print('{}/{} ({:.2f}%) MIDI files successfully processed'.format(ok_cnt, all_cnt, ok_cnt / all_cnt * 100))