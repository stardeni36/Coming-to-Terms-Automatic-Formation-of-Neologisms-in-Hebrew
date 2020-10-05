from other.naive_shoresh_mishkal_combine import shoresh_mishkal_combine_with_spacials,\
    HEB_LETTERS_START

NUM_OF_FEATURES = 12

def get_gzarot_prefix(shoresh):
    gzarot_prefix = ""
        
    if len(shoresh) == 4: # ×ž×¨×•×‘×¢×™×�
        gzarot_prefix = "MRBA NPG NPA NPN NPY NAG NAVY NLG NLYH NLA NKFL"
    else: 
        gzarot_prefix += "NMRBA "
        if shoresh[0] in ['×”', '×—', '×¢', '×�', '× ', '×™']: 
            # ×¤"×’ (×”,×—,×¢)
            if shoresh[0] in ['×”', '×—', '×¢']:
                gzarot_prefix += "PG "
            else:
                gzarot_prefix += "NPG "
            # ×¤"×� (×�)
            if shoresh[0] == '×�':
                gzarot_prefix += "PA "
            else: 
                gzarot_prefix += "NPA "
            # ×—×¤"×  (× )
            if shoresh[0] == '× ':
                gzarot_prefix += "PN "
            else:
                gzarot_prefix += "NPN "
            # ×¤"×™ (×™)
            if shoresh[0] == '×™':
                gzarot_prefix += "PY "
            else:
                gzarot_prefix += "NPY "
        else:
            gzarot_prefix += "NPG NPA NPN NPY "
            
        if shoresh[1] in ['×�','×”','×—','×¢','×™','×•']:
            # ×¢"×’ (×�,×”,×—,×¢)
            if shoresh[1] in ['×�','×”','×—','×¢']:
                gzarot_prefix += "AG "
            else:
                gzarot_prefix += "NAG "
            # ×¢×•"×™ (×™,×•)
            if shoresh[1] in ['×™', '×•']:
                gzarot_prefix += "AVY " 
            else:
                gzarot_prefix += "NAVY "
        else:
            gzarot_prefix += "NAG NAVY "
                  
        if shoresh[2] in ['×”','×—','×¢','×™','×�']:
            # ×œ"×’ (×”,×—,×¢)
            if shoresh[2] in ['×”','×—','×¢']:
                gzarot_prefix += "LG "
            else:
                gzarot_prefix += "NLG "
            # ×œ×™"×” (×™,×”)
            if shoresh[2] in ['×™','×”']:
                gzarot_prefix += "LYH "
            else: 
                gzarot_prefix += "NLYH " 
            # ×œ"×� (×�)
            if shoresh[2] == '×�':
                gzarot_prefix += "LA "
            else: 
                gzarot_prefix += "NLA "
        else: 
            gzarot_prefix += "NLG NLYH NLA "   
        # ×›×¤×•×œ×™×�
        if shoresh[1] == shoresh[2]:
            gzarot_prefix += shoresh[1] # "KFL"
        else: 
            gzarot_prefix += "NKFL"
    
    return gzarot_prefix
    
        
def groni_tag(shoresh, letter_idx):
    if (shoresh[letter_idx] in ['×�','×”','×—','×¢']):
        return 1
    return 0


def g_alef_tag(shoresh, letter_idx):
    if (shoresh[letter_idx] == '×�'):
        return 1
    return 0


def g_hey_tag(shoresh, letter_idx):
    if(shoresh[letter_idx] == '×”'):
        return 1
    return 0


def g_het_tag(shoresh, letter_idx):
    if(shoresh[letter_idx] == '×—'):
        return 1
    return 0


def g_ain_tag(shoresh, letter_idx):
    if(shoresh[letter_idx] == '×¢'):
        return 1
    return 0


def first_letter_noon(shoresh, letter_idx):
    if(shoresh[0] == '× ' and letter_idx==0):
        return 1
    return 0


def yood(shoresh, letter_idx):
    if(shoresh[letter_idx] == '×™'):
        return 1
    return 0


def second_letter_vav(shoresh, letter_idx):
    if(shoresh[1] == '×•' and letter_idx==1):
        return 1
    return 0


def repeated_shoresh_letter(shoresh, letter_idx):
    if shoresh[1]==shoresh[2] and (letter_idx==1 or letter_idx==2):
        return 1
    return 0

def mrba(shoresh):
    if len(shoresh)==4:
        return 1
    return 0


# input: shoresh. output: binary array of length: 12. 
# 12 features: nikud? shoresh letter? groni? g-alef? g-hey? g-het? g-ain? first letter noon? yood? second letter vav? repeated shoresh letter? mrba?
def get_gzarot_tag_vector(shoresh, letter_idx):
    gzarot_vector = [0,1] # no nikud, shoresh letter
    gzarot_vector += [groni_tag(shoresh, letter_idx)]
    gzarot_vector += [g_alef_tag(shoresh, letter_idx)]
    gzarot_vector += [g_hey_tag(shoresh, letter_idx)]
    gzarot_vector += [g_het_tag(shoresh, letter_idx)]
    gzarot_vector += [g_ain_tag(shoresh, letter_idx)]
    gzarot_vector += [first_letter_noon(shoresh, letter_idx)]
    gzarot_vector += [yood(shoresh, letter_idx)]
    gzarot_vector += [second_letter_vav(shoresh, letter_idx)]
    gzarot_vector += [repeated_shoresh_letter(shoresh, letter_idx)]
    gzarot_vector += [mrba(shoresh)]
    return gzarot_vector
    
    
def shoresh_mishkal_to_gzarot_vecs(shoresh, mishkal):
    word, shoresh_letter_places = shoresh_mishkal_combine_with_spacials(shoresh, mishkal, return_shoresh_letter_places=True)
    gzarot_vecs = []
    
    for i in range(len(shoresh_letter_places)):
        vec = []
        if shoresh_letter_places[i] == 0:
            vec = [0]*NUM_OF_FEATURES
            if ord(word[i]) < HEB_LETTERS_START:
                vec[0] = 1 # the character is a nikud character
        else:
            vec = get_gzarot_tag_vector(shoresh, shoresh_letter_places[i]-1)
    
#         print(vec)    
        gzarot_vecs += [vec]
     
#     print(shoresh)
#     print(mishkal)
#     print(word)
#     print(shoresh_letter_places)
#     print("")
    return gzarot_vecs
    

def get_gzarot_vectors_str(gzarot_vectors):
    
    vec_str = ""
    
    for vector in gzarot_vectors:
        vector = [str(item) for item in vector]
        vec_str += ' '.join(vector)
        vec_str += ' 2 '
        
    return vec_str[:-3]


def create_index_to_shoresh_gzarot_mishkal_dictionary(file_path):
    shorashim = []
    mishkalim = {}
    
    index2shoresh = {}  # index -> 'shoresh', 'index_to_shoresh_gzarot'
    
    i=-1
    
    with open(file_path, 'r', encoding='utf8') as f:    
        for line in f:
            spltd = line.split(",")
            shoresh = spltd[0].strip()
            shorashim += [(i, shoresh)]
            mishkal = spltd[1].strip()
            mishkalim[i] = mishkal
            i+=1
            
    shorashim = shorashim[1:]
#     print(len(shorashim))
    
    for i, shoresh in shorashim:
        index2shoresh[i] = {}
        index2shoresh[i]['shoresh'] = shoresh
        mishkal = mishkalim[i]
        index2shoresh[i]['mishkal'] = mishkal
        
        print(i)
        print(shoresh)
        print(mishkal)
        
        gzarot_vectors = shoresh_mishkal_to_gzarot_vecs(shoresh, mishkal)
        gzarot_prefix = get_gzarot_prefix(shoresh)
        
        index2shoresh[i]['gzarot_prefix'] = gzarot_prefix
        index2shoresh[i]['gzarot_vectors'] = gzarot_vectors
        index2shoresh[i]['gzarot_vectors_string'] = get_gzarot_vectors_str(gzarot_vectors)
        
#         print(get_gzarot_vectors_str(gzarot_vectors))
    
    return index2shoresh


# if gzarot_prefix=True inserts gzarot prefixes to file, else: inserts gzarot feature vectors to file.
def insert_gzarot_info_to_file(file_path, target_path, index_to_shoresh_dict, gzarot_prefix=True):
    first = True
    with open(file_path, 'r', encoding='utf8') as f:
        with open(target_path, 'w', encoding='utf8') as f2:
            for line in f:
                if first: 
                    f2.write("index,gzarot_prefix, gzarot_concat,shoresh,combine_result,word\n")
                    first = False
                else:
                    spltd = line.split(",")
                    index = int(spltd[0])
                    combine_result = spltd[1]
                    word = spltd[2]
                    gzarot_concat = index_to_shoresh_dict[index]['gzarot_vectors_string']
                    gzarot_prefix = index_to_shoresh_dict[index]['gzarot_prefix']
                    shoresh = index_to_shoresh_dict[index]['shoresh']
                    shoresh = " ".join(list(shoresh))
#                     mishkal = index_to_shoresh_dict[index]['mishkal']
                    f2.write(str(index) + "," + gzarot_prefix +"," + gzarot_concat + "," + shoresh + "," + combine_result + "," + word)


if __name__ == '__main__':
    # for index in index_to_shoresh_dict:
    #     print(index_to_shoresh_dict[index]['shoresh'])
    #     print(index_to_shoresh_dict[index]['gzarot'])
    #     print(index_to_shoresh_dict[index]['mishkal'])
    #     print("")
    
    index2shoresh = create_index_to_shoresh_gzarot_mishkal_dictionary('combine_results_vs_real.csv')
    insert_gzarot_info_to_file("test.csv", "test_new5.csv", index2shoresh, gzarot_prefix=False)
    insert_gzarot_info_to_file("val.csv", "val_new5.csv", index2shoresh, gzarot_prefix=False)
    insert_gzarot_info_to_file("train.csv", "train_new5.csv", index2shoresh, gzarot_prefix=False)
    
    
    
    

            
            