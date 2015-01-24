for i in $(seq 1 51):
	do 
		echo $i
		echo $i.jpg
		./c images/$i.jpg
	done
