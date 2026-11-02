from copy import deepcopy


def text_input(item, transcript, history_transcripts):
    """Only explicitly selected transcripts, never a Reference object."""
    if not isinstance(transcript, str): raise ValueError("transcript must be text")
    history, user_index = [], 0
    for turn in item.history:
        if turn['role'] == 'user':
            if user_index >= len(history_transcripts): raise ValueError('missing history transcript')
            history.append({'role':'user','text':history_transcripts[user_index]})
            user_index += 1
        else: history.append(deepcopy(turn))
    if user_index != len(history_transcripts): raise ValueError('extra history transcript')
    return {'request':transcript, 'context':deepcopy(item.context),
            'history':history, 'tools':deepcopy(item.tools)}
