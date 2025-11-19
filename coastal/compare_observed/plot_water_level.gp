obv=ARG1
sim=ARG2

print 'obv = ', obv
print 'sim = ', sim

id = system( 'basename ' . obv . ' _obv.txt' )
print 'id = ', id

if ( ARGC == 3 ){
  out=ARG3
  set term out size 900, 600 enhanced
  set output id . "_gnuplot.png"
}
else{
 #set term x11
 set term qt
}
set ylabel "Elevation ASL (m)"

set xdata time

set format x "%d %H:%M"
set xtics rotate by 45 right

set title id
set timefmt "%Y%m%d_%H:%M"
plot \
    obv using 1:2 t "obs" with points pt 6 ps 1, \
    sim using 1:2 t "sim" with lines lw 4 lc rgb 'black'
