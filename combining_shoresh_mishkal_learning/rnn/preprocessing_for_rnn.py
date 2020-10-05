import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from eliezer_bot_yehuda.shoresh_mishkal_combine import shoresh_mishkal_combine_with_spacials,\
    final_letter_to_regular_letter


def create_csv_file_shoresh_mishkal_hewiktionary():
    # reading dictionaries
    with open('mishkalim/jsons/heb_word2mishkal.json', 'r') as f:
        mishkalim = json.load(f)

    with open('shorashim/jsons/shorashim_cleaned.json', 'r') as f:
        shorashim = json.load(f)

    # combine all data to one file
    # new keys contain only words with both shoresh and mishkal (intersection)
    new_keys = set(mishkalim.keys()).intersection(set(shorashim.keys()))
    print('number of new keys: ' + str(len(new_keys)))

    # write to csv file
    with open('data/shoresh_mishkal_stav_data.csv', 'w') as f:
        f.write('word,shoresh,mishkal\n')
        for key in new_keys:
            shoresh = shorashim[key]
            mishkal = mishkalim[key]
            if len(mishkal) > 1:
                raise Warning('more than 1 mishkal')
            else:
                f.write(key + ',' + shoresh + ',' + str(mishkal[0]) + '\n')


def save_cleaned():
    filename = 'data/shoresh_mishkal_stav_data.csv'
    df = pd.read_csv(filename)
    # clean from shoresh
    df['shoresh'] = [final_letter_to_regular_letter(i) for i in df['shoresh']]
    # clean shoresh
    df['shoresh'] = df['shoresh'].replace(r'י/ה', 'י', regex=True)
    df['shoresh'] = df['shoresh'].replace(r'ה/י', 'י', regex=True)
    df['shoresh'] = df['shoresh'].replace(chr(1473), "", regex=True)
    df['shoresh'] = df['shoresh'].replace(chr(1474), "", regex=True)
    # write to file
    df.to_csv('data/shoresh_mishkal_stav_data_cleaned.csv', index=0)
    # TODO: two words cleaned manually: 2098, 2929
    # TODO manually remove /


def create_datasets_with_func_output():
    """
    create csv file with shoresh, mishkal, input and output to rnn model
    the input comes from naive function combining shoresh and mishkal
    named "shoresh_mishkal_combine_with_spacials"
    """
    # TODO: for this to work i fixed some nonsense shorashim
    filename = 'data/shoresh_mishkal_stav_data_cleaned_prev.csv'
    df = pd.read_csv(filename)
    with open('data/combine_results_vs_real2.csv', 'w') as f:
        f.write('shoresh,mishkal,combine_result,word\n')
        for i in range(len(df)):
            row = df.iloc[i]
            shoresh = row['shoresh']
            mishkal = row['mishkal']
            word = row['word']
            res = shoresh_mishkal_combine_with_spacials(shoresh, mishkal)
            if res:
                f.write(shoresh + ',' + mishkal + ',' + res + ',' + word + '\n')
                # TODO - i manually erased plurals and fixed agavim's mishkal
                # TODO there is no 2 in name


def letter_to_category():  # now suitable for the rnn case
    """
    create and save dictionaries from index to letter and from letter to index
    """
    filename = '../cat_nb/data/shoresh_mishkal_stav_data_cleaned_prev.csv'  # previously without prev
    df = pd.read_csv(filename)
    words = df['word']
    txt = ''
    for w in words:
        txt = txt.strip() + w
    txt_uniqs = list(set(txt))
    dict_letter_to_cat = {i: j+3 for j, i in enumerate(txt_uniqs)}  # non zero  j+1 originally
    dict_letter_to_cat.update({"PAD": 0, "SOS": 1, "EOS": 2})
    with open('letter_to_cat_rnn.json', 'w') as outfile:
        json.dump(dict_letter_to_cat, outfile)
    reverse = {j:i for i,j in dict_letter_to_cat.items()}
    with open('cat_to_letter_rnn.json', 'w') as outfile:
        json.dump(reverse, outfile)


def split_dataset():
    """
    save train, val and test in csv
    """
    df = pd.read_csv('combine_results_vs_real.csv')
    df_x = df['combine_result']
    df_y = df['word']
    X_train_val, X_test, y_train_val, y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    train_set = pd.DataFrame({'combine_result': X_train, 'word': y_train})
    val_set = pd.DataFrame({'combine_result': X_val, 'word': y_val})
    test_set = pd.DataFrame({'combine_result': X_test, 'word': y_test})
    train_set.to_csv('data/train.csv')
    val_set.to_csv('data/val.csv')
    test_set.to_csv('data/test.csv')


# analysis #


def check_new_dataset():
    """
    how many of the inputs and outputs are differnet?
    """
    filename = 'combine_results_vs_real.csv'
    df = pd.read_csv(filename)
    a = np.where(df['word'] != df['combine_result'])
    print(a[0].size)


def datasets_analysis(phase):
    """
    check equal and differnet input and target
    for specific dataset
    :param phase: train, val or test
    """
    df = pd.read_csv('data/' + phase + '.csv')
    a = np.where(df['word'] == df['combine_result'])
    print(a[0].size)
    a = np.where(df['word'] != df['combine_result'])
    print(a[0].size)


def combine_train_and_val(train_path, val_path):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    train_val = pd.concat([train, val])  # TODO: make sure its the right axis
    train_val.to_csv('data/train_val_new5.csv')


if __name__ == '__main__':
    # split_dataset()
    # datasets_analysis('train')
    # datasets_analysis('val')
    # datasets_analysis('test')
    # letter_to_category()
    combine_train_and_val('data/train_new5.csv', 'data/val_new5.csv')
    # pass


