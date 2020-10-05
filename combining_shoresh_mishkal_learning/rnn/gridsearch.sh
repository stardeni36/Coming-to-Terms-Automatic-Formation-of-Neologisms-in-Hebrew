for lr in $(cat config/lr)
do
	for bs in $(cat config/bs)
	do
		for hd in $(cat config/hd)
		do
			for t in $(cat config/teacher)
			do
				for d in $(cat config/do)
				do
					python3 rnn_for_combine_padded_batches.py --learning_rate=${lr} --batch_size=${bs} --hidden_size=${hd} --teacher_forcing=${t} --dropout_p=${d}
				done 		
			done
		done
	done
done
