pval() { "$@" ; echo $? ; }
for m in cvqr pyzbar; do
	O=out_noises_$m.txt
	echo > $O
	for i in `seq 1 2 50`; do
		CX=0
		for c in `seq 20`; do
			cat img/qrwiki.png | SRAND=$c plambda "randg $i * +"| python fiducials.py $m 2>/dev/null && CX=$[CX+1]
		done
		echo $i $CX >> $O
	done
done
gnuplot -persist -e 'plot [0:11] [-2:22] "out_noises_cvqr.txt" w lines, "out_noises_pyzbar.txt" w lines'
