
for sfe in "001" "010" "030"; do
    for ndens in "1e4_BE"; do
        for mCloud in "1e7"
	   do
		start_time=$(date +%s)
		start_date=$(date)
		filename="${mCloud}_sfe${sfe}_n${ndens}"
		
		echo "========================================"
    	        echo "Running $filename... at $start_date"
		echo "========================================"
		
		python3 run.py param/${filename}.param > txt/${filename}.txt &
	   done
    done
done
