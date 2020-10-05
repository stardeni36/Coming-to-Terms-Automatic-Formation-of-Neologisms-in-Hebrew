import numpy as np
import operator
from other.naive_shoresh_mishkal_combine import clean_word_from_nikud,\
    HEB_LETTERS_START


# some nikud normalizations. For example, changes all tzeres and hataf segols to segols, changes all kamatz / hataf patach/ hataf kamatz to hataf.    
def string_preprocessing(str):
    last_shva = False
    processed = ""
    for i in range(len(str))[::-1]:
        if i==len(str)-1 and ord(str[i]) == 1456: # שווא
            if not (i>0 and (str[i-1])=="ך"):
                processed = str[i] + processed
        elif str[i]=="י":
            if not(i!=len(str)-1 and ord(str[i+1])>=HEB_LETTERS_START and i>0 and ord(str[i-1])==1460): # 1460 = חיריק
                processed = str[i] + processed
        elif (ord(str[i])==1458 or ord(str[i])==1459 or ord(str[i])==1464): # 1458, 1459, 1460 - חטף פתח / קמץ, קמץ
            processed = chr(1463) + processed # 1463 - פתח
        elif (ord(str[i])==1457 or ord(str[i])==1461): # 1457, 1461 - חטף סגול, צירה
            processed = chr(1462) + processed # 1462 - סגול
        elif (ord(str[i])==1473 or ord(str[i])==1474): # 1473/4 - shin/sin
            if not (i>0 and (str[i-1])=="ש"):
                processed = str[i] + processed
        elif (ord(str[i])==1468): # 1468 - שורוק / דגש / מפיק
            if not (i>0 and str[i-1]=="ו"):
                processed = str[i] + processed
        elif (str[i]=="ו" and i!=len(str)-1 and ord(str[i+1])==1468 and i>0 and (ord(str[i-1])>=HEB_LETTERS_START or ord(str[i-1])==1473 or ord(str[i-1])==1474 or ord(str[i-1])==1468)):
            processed = chr(1467) + processed # 1467 - קובוץ
        elif (str[i]=="ו"):
            if not(i!=len(str)-1 and ord(str[i+1])==1465):
                processed = str[i] + processed
        else:
            processed = str[i] + processed
             
#     print(processed)
    return processed


def edit_distance(str1, str2):

    str1 = string_preprocessing(str1)
    str2 = string_preprocessing(str2)
    
    arr = np.zeros((len(str1)+1, len(str2)+1))
    
    # recursion base:
    for i in range(1, len(str1)+1):
        arr[i][0] = i
    
    for j in range(1, len(str2)+1):
        arr[0][j] = j

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1]==str2[j-1]:
                arr[i][j] = arr[i-1][j-1]
            else:
                arr[i][j] = min(arr[i-1][j-1]+1, arr[i-1][j]+1, arr[i][j-1]+1)
    
    return (arr[len(str1)][len(str2)])
        

def edit_distance_no_nikud(str1, str2):
    str1 = clean_word_from_nikud(str1)
    str2 = clean_word_from_nikud(str2)
    return edit_distance(str1, str2)


def clean_word_from_regular_letters(word): # keep only the nikud letter series.
    cleaned = ""
    for letter in word:
        if (ord(letter)<HEB_LETTERS_START) or (letter==' ' or letter=='-'):
            cleaned += letter
    return cleaned

 
def edit_distance_nikud_only(str1, str2):
    str1 = clean_word_from_regular_letters(str1)
    str2 = clean_word_from_regular_letters(str2)
    return edit_distance(str1, str2)
    
# string_preprocessing(A)
# edit_distance(A, B)
def results_accuracy(file_name):
    same = 0
    same_after_processing = 0
    edit_distance_sum = 0
    edit_distance_no_nikud_sum = 0
    edit_distance_nikud_only_sum = 0
    total = 0
    total_edits = 0
    ed_dict = {}
    
    with open(file_name, 'r', encoding='utf8') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
            else: 
                total += 1
                spltd = line.split(',')
                combined = spltd[0].strip()
                predicted = spltd[1].strip()
                target = spltd[2].strip()
                if predicted == target:
                    same += 1
                else:
                    pred_proc = string_preprocessing(predicted)
                    tar_proc = string_preprocessing(target)
                    if pred_proc == tar_proc:
                        same_after_processing += 1
                    else: 
                        total_edits += 1
                        ed = edit_distance(pred_proc, tar_proc)
                        ed_no_nikud = edit_distance_no_nikud(pred_proc, tar_proc)
                        ed_nikud_only = edit_distance_nikud_only(pred_proc, tar_proc)
                        edit_distance_sum += ed
                        edit_distance_no_nikud_sum += ed_no_nikud
                        edit_distance_nikud_only_sum += ed_nikud_only
                        if int(ed) not in ed_dict:
                            ed_dict[ed] = []
                        ed_dict[ed] += [(combined, predicted, target)]
                            
                            
    print("total: ", total)
    print("same: ", same)
    print("same after processing: ", same_after_processing)
    print("edit distance sum: ", edit_distance_sum)
    print("edit distance avg: ", edit_distance_sum/total)
    print("total edits: ", total_edits)
    print("edit distance avg (for mistakes only): ", edit_distance_sum/total_edits)
    print("edit distance avg (regular letters only): ", edit_distance_no_nikud_sum/total_edits)
    print("edit distance avg (nikud only): ", edit_distance_nikud_only_sum/total_edits)
    print("accuracy: ", (same + same_after_processing)/(1.0*total))


                
            
                
            
            
    
    

  


