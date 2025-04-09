
for sfe in "001" "010" "030"; do
    for ndens in "1e4"; do
        for mCloud in "1e6"
	   do
		start_time=$(date +%s)
		start_date=$(date)
		filename="${mCloud}_sfe${sfe}_n${ndens}"
		
		echo "========================================"
    	        echo "Running $filename... at $start_date"
		echo "========================================"
		
		python3 run.py param/${filename}.param > txt/${filename}.txt &
	    	
		end_time=$(date +%s)
		end_date=$(date)
        	runtime=$((end_time - start_time))

		echo "========================================"
		echo "Completed $filename at $end_date"
		echo "Total run time: ${runtime} seconds"
		echo "========================================"
	   done
    done
done
