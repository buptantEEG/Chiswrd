import pickle
import numpy as np

event_id = {"fixation": 13,
        "close_eyes": 15,
        "concentrate_data": 21,
        "action_data": 22,
        "rest_data": 23,
        "我": 30,
        "你": 31,
        "他": 32,
        "上": 33,
        "下": 34,
        "左": 35,
        "右": 36,
        "是": 37,
        "水": 38,
        "饭": 39}

def np_append(event_np, sample, id):
    event = np.zeros((3), dtype=int)
    event[0] = sample
    event[2] = id
    event_np = np.vstack((event_np, event))
    return event_np

def event_generate(path):
    file = open(path + 'data_list.pkl', 'rb')
    data = pickle.load(file)
    # print((data[2].keys()))
    # load all raw data
    eye_open = data[0]['data']
    eye_close = data[1]['data']
    all_data = np.hstack((eye_open, eye_close))
    for trail in range(2, len(data)):
        all_data = np.hstack((all_data, data[trail]['concentrate_data']))
        all_data = np.hstack((all_data, data[trail]['cue_data']))
        all_data = np.hstack((all_data, data[trail]['action_data']))
        all_data = np.hstack((all_data, data[trail]['rest_data']))
    print(all_data.shape)

    # generate event
    event_np = np.array([-1, 0, 65536], dtype=int)
    sample = 0
    id = 13
    # fixation
    event_np = np_append(event_np, sample, id)
    sample += data[0]['data'].shape[-1]
    # close_eyes
    id = 15
    event_np = np_append(event_np, sample, id)
    sample += data[1]['data'].shape[-1]
    # first trial

    for i in range(2, len(data)):
        # concentrate_data
        id = 21
        event_np = np_append(event_np, sample, id)
        sample += data[i]['concentrate_data'].shape[-1]
        #  'cue_data', 'action_data'
        if data[i]['label'] == '我':
            # cue_data
            id = 30
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '你':
            id = 31
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '他':
            id = 32
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '上':
            id = 33
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '下':
            id = 34
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '左':
            id = 35
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '右':
            id = 36
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '是':
            id = 37
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '水':
            id = 38
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        elif data[i]['label'] == '饭':
            id = 39
            event_np = np_append(event_np, sample, id)
            sample += data[i]['cue_data'].shape[-1]
        # action_data
        id = 22
        event_np = np_append(event_np, sample, id)
        sample += data[i]['action_data'].shape[-1]
        # rest_data
        id = 23
        event_np = np_append(event_np, sample, id)
        sample += data[i]['rest_data'].shape[-1]
        # event_np.append(event)
    event_np = np.array(event_np)
    print(event_np.shape)
    return event_np, all_data

if __name__=='__main__':
    path = 'xxx'
    event = event_generate(path)