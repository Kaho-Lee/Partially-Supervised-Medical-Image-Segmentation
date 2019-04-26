for i in {100..160}
do
    echo "output: $i"
    convert -size 2048x2048 -depth 16 -endian MSB -normalize gray:JSRT/All247images/JPCLN$i.IMG JSRT/All247/JPCLN$i.png
done
