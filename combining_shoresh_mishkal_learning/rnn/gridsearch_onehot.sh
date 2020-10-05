for lr in $(cat config/lr)
do
	for bs in $(cat config/bs)
	do
		for t in $(cat config/teacher)
		do
			for d in $(cat config/do)
			do
				python3 rnn_before_not_embedded.py --learning_rate=${lr} --batch_size=${bs} --teacher_forcing=${t} --dropout_p=${d}
			done 		
		done
	done
done
