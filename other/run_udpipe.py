# -*- coding: utf-8 -*-
from time import time
from ufal.udpipe import Model, Pipeline, ProcessingError
import os

dirname = os.path.dirname(os.path.abspath(__file__))


def udpipe(lines, model_name, batch_size=256):
    """
    Parse text to Universal Dependencies using UDPipe.
    :param lines: list of strings
    :param model: the model itself (after loading)
    :return: the parsed data
    """
    
    pipeline = Pipeline(model_name, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    batches = [lines[x:x + batch_size]
               for x in range(0, len(lines), batch_size)]
    results = []
    
    tag_data = []
    
    for i, batch in enumerate(batches):
        text = "".join(batch)
        error = ProcessingError()
        num_tokens = sum(1 for l in batch if l)
        # print("Running UDPipe on %d tokens, batch %d " %
        #       (num_tokens, i))
        start = time()
        processed = pipeline.process(text, error)
        duration = time() - start
#         print("Done (%.3fs, %.0f tokens/s)" %
#               (duration, num_tokens / duration if duration else 0))
        
        if error.occurred():
            raise RuntimeError(error.message)
        
#         print (processed.splitlines())
        tag_data = processed.splitlines() 
        results.extend(processed.splitlines())
        
    return tag_data 

def is_optional_nn_suggestion(tag_data):
    if (tag_data[3].split('\t')[3]=='NOUN' and tag_data[4].split('\t')[3]=='NOUN'):
        return True
    return False

def no_gender_problematic_suggestion(tag_data):
    if ('Gender=Masc' in tag_data[3] and 'Gender=Fem' in tag_data[4]):
        return False
    elif ('Gender=Fem' in tag_data[3] and 'Gender=Masc' in tag_data[4]):
        return False
    return True

def no_uncertain_label(tag_data):
    to_search = 'HebSource=ConvUncertainLabel' 
    if (to_search not in tag_data[3] and to_search not in tag_data[4]):
        return True
    return False

def no_number_problematic_suggestion(tag_data):
    if ('Number=Plur' in tag_data[3] and 'Number=Sing' in tag_data[4]):
        return False
    elif ('Number=Sing' in tag_data[3] and 'Number=Plur' in tag_data[4]):
        return False
    return True

def is_optional_suggestion(udpipe_model, suggestion, check_smixut=True):
    suggestion = [suggestion]
    tag_data = udpipe(suggestion, udpipe_model)
    if (check_smixut):
        if(is_optional_nn_suggestion(tag_data) and contains_smixut(tag_data)):
            return True
#     if (no_gender_problematic_suggestion(tag_data) and no_uncertain_label(tag_data) and no_number_problematic_suggestion(tag_data)): # TODO: think. adj... 
#         return True
    
    return False

def contains_smixut(tag_data):
    if ('compound:smixut' in tag_data[4]):
        return True
    return False
    
