for sfe in "030"; do
    for ndens in "1e4"; do
        for mCloud in "1e5"; do
            for PISM in "0" "1e4" "1e5" "1e6"; do
            		start_time=$(date +%s)
            		start_date=$(date)
            		filename="${mCloud}_sfe${sfe}_n${ndens}_PISM${PISM}"
            		
            		echo "========================================"
                	        echo "Running $filename... at $start_date"
            		echo "========================================"
            		
            		python3 run.py param/${filename}.param > txt/${filename}.txt &
             done   	
	 done
    done
done

