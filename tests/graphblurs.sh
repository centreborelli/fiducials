pval() { "$@" ; echo $? ; }
for m in cvqr pyzbar; do
	O=out_blurs_$m.txt
	echo > $O
	for i in `seq 6 0.1 8`; do
		cat img/qrwiki.png | blur g $i | echo $i `pval python fiducials.py $m 2>/dev/null` >> $O
	done
done
gnuplot -e 'plot [5:9] [-1:2] "out_blurs_cvqr.txt" w lines, "out_blurs_pyzbar.txt" w lines'
