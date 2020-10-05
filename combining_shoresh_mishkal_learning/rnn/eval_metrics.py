import pandas as pd

# df = pd.read_csv('results_eval_before_nikud_standardization.csv')
df = pd.read_csv('results_eval_teacher=0.5.csv')

acc = sum(df['predicted'] == df['target'])/ len(df)

print('accuracy (predicted = target): ' + str(acc))

per_same = sum(df['input'] == df['target'])/ len(df)

print('percentage input = target: ' + str(per_same))

# divide to different cases
# case 1: input, predicted and target are the same, not so interesting
m1 = df[(df['predicted'] == df['target']) & (df['target'] == df['input'])]
print(len(m1))
# case 2: input is different from target, but predicted = target (success)
m2 = df[(df['predicted'] == df['target']) & (df['target'] != df['input'])]
print(len(m2))
# case 3: input equal to target, but the prediction is different - model messed it up
m3 = df[(df['predicted'] != df['target']) & (df['target'] == df['input'])]
print(len(m3))
# case 4: the input is different from target, and the model didn't fix it
m4 = df[(df['predicted'] != df['target']) & (df['target'] != df['input'])]
print(len(m4))

# save results in separate files

m1.to_csv('m1.csv')
m2.to_csv('m2.csv')
m3.to_csv('m3.csv')
m4.to_csv('m4.csv')