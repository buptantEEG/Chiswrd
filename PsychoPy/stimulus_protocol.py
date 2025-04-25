from psychopy import visual, core, event, sound
import random
import datetime

def generate_filename(sub_id, session_id, per_word_repeat_time, name):

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    filename = f"{sub_id}_session{session_id}_repeat{per_word_repeat_time}_{current_time}_{name}.txt"
    return filename
 
sub_id = 'xxx'
session_id = str(1)
per_word_repeat_time = 10
output_file = generate_filename(sub_id = sub_id, session_id = session_id, per_word_repeat_time = per_word_repeat_time, name = 'timestamps')
questionnaire_file = generate_filename(sub_id = sub_id, session_id = session_id, per_word_repeat_time = per_word_repeat_time, name = 'questionnaire_results')
sequence_file = generate_filename(sub_id = sub_id, session_id = session_id, per_word_repeat_time = per_word_repeat_time, name = 'stimuli_sequence')
print('output_file:', output_file)
print('questionnaire_file:', questionnaire_file)
print('sequence_file:', sequence_file)

win = visual.Window(size=(800, 600), color="black", units="pix", fullscr=True)
text = visual.TextStim(win, text="按空格键进入下一阶段", color="white", height=50, pos=(0, 0)) 
concentrate_circle = visual.Circle(win, radius=50, fillColor="white", lineColor="white", pos=(0, 0)) 
next_phase_text = visual.TextStim(win, text='按空格进行下一阶段', color="white", height=50, pos=(0, 0)) # 创建文本提示
rest_text = visual.TextStim(win, text='休息', color="white", height=100, pos=(0, 0), alignText='center')
stimuli_words = ['我', '你', '他', '上', '下', '左', '右', '是', '水', '饭']
stimuli_list = stimuli_words * per_word_repeat_time
random.shuffle(stimuli_list)
timestamps = [] 
questionnaire_results = []  

def check_for_exit():
    keys = event.getKeys(keyList=['escape']) 
    if 'escape' in keys:
        win.close()
        core.quit()

text.draw()
win.flip()
keys = event.waitKeys(keyList=['space'])

fixation = visual.TextStim(win, text='+', color="white", height=300, pos=(0, 0))  
fixation_start = datetime.datetime.now()  
for _ in range(600): 
    fixation.draw()
    win.flip()
    core.wait(0.1) 
    check_for_exit()  
fixation_end = datetime.datetime.now()  
timestamps.append({'event': 'fixation', 'start': fixation_start, 'end': fixation_end})

text.draw() 
win.flip() 
keys = event.waitKeys(keyList=['space'])

close_eyes = visual.TextStim(win, text='闭眼', color="white", height=200, pos=(0, 0)) 
close_eyes.draw() 
win.flip() 
keys = event.waitKeys(keyList=['space'])

close_eyes_start = datetime.datetime.now() 
for _ in range(600):
    win.flip()
    core.wait(0.1)  
    check_for_exit()  
close_eyes_end = datetime.datetime.now()  
timestamps.append({'event': 'close_eyes', 'start': close_eyes_start, 'end': close_eyes_end})

while True:
    next_phase_text.draw()
    win.flip()  
    keys = event.waitKeys(keyList=['space', 'escape']) 
    if 'escape' in keys:
        core.quit() 
    elif 'space' in keys:
        break 

for word in stimuli_list:
    concentrate_start = datetime.datetime.now()  
    concentrate_circle.draw()
    win.flip()
    for i in range(200): 
        check_for_exit()
        core.wait(0.01)
    concentrate_end = datetime.datetime.now() 

    cue_start = datetime.datetime.now() 
    cue_text = visual.TextStim(win, text=word, color="white", height=100, pos=(0, 0), alignText='center')
    cue_text.draw()
    win.flip()
    for i in range(200):
        check_for_exit()
        core.wait(0.01)
    cue_end = datetime.datetime.now() 

    action_start = datetime.datetime.now()
    win.flip() 
    for i in range(400):
        check_for_exit()
        core.wait(0.01)
    action_end = datetime.datetime.now() 

    rest_start = datetime.datetime.now()
    rest_text.draw()
    win.flip()
    for i in range(200):
        check_for_exit()
        core.wait(0.01)
    rest_end = datetime.datetime.now()

    timestamps.append({
        'event': 'trial',
        'word': word,
        'concentrate_start': concentrate_start,
        'concentrate_end': concentrate_end,
        'cue_start': cue_start,
        'cue_end': cue_end,
        'action_start': action_start,
        'action_end': action_end,
        'rest_start': rest_start,
        'rest_end': rest_end
    })

    if random.choice([True, False, False, False]):
        correct_word = word
        incorrect_word = random.choice([w for w in stimuli_words if w != word])
        options = [correct_word, incorrect_word]
        random.shuffle(options) 

        question_text = visual.TextStim(win, text="请选择刚刚展示的单词:", color="white", height=50, pos=(0, 200), alignText='center')

        left_option = visual.TextStim(win, text=options[0], color="white", height=40, pos=(-200, 100), alignText='center')

        right_option = visual.TextStim(win, text=options[1], color="white", height=40, pos=(200, 100), alignText='center')


        question_text.draw()
        left_option.draw()
        right_option.draw()
        win.flip()

        while True:
            keys = event.waitKeys(keyList=['left', 'right', 'escape'])

            if 'escape' in keys:
                print("用户选择退出")
                core.quit() 

            if keys[0] in ['left', 'right']:
                break 

        chosen_option = options[0] if keys[0] == 'left' else options[1]

        questionnaire_results.append({
            'word': word,
            'options': options,
            'chosen_option': chosen_option,
            'correct': chosen_option == word
        })


with open(output_file, "w") as f:
    for entry in timestamps:
        f.write(f"{entry}\n")

with open(questionnaire_file, "w") as f:
    for result in questionnaire_results:
        f.write(f"{result}\n")

with open(sequence_file, "w") as f:
    for word in stimuli_list:
        f.write(word + "\n")


win.close()
core.quit()






