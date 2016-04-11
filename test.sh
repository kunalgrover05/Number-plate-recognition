for i in $(seq 1 51):
	do 
		echo $i
		echo $i.jpg
		./a.out images/$i.jpg
	done
