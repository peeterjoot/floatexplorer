./testbin/e4m3.sh | diff -pU100 expected/e4m3.all.txt -
./testbin/e5m2.sh | diff -pU100 expected/e5m2.all.txt -
./floatexplorer --e4m3 3 | diff -pU100 expected/e4m3.txt -
./floatexplorer --e5m2 3 | diff -pU100 expected/e5m2.txt -
./floatexplorer --spe | diff -pU100 expected/float.special.txt -
./floatexplorer --spe --double | diff -pU100 expected/double.special.txt -
./floatexplorer --spe --e4m3 | diff -pU100 expected/e4m3.special.txt -
./floatexplorer --spe --e5m2 | diff -pU100 expected/e5m2.special.txt -
./floatexplorer --spe --fp16 | diff -pU100 expected/fp16.special.txt -
./floatexplorer --spe --bf16 | diff -pU100 expected/bf16.special.txt -

exit
# review:
./testbin/bf16.sh | diff -pU100 expected/bf16.all.txt -
./testbin/fp16.sh | diff -pU100 expected/fp16.all.txt -
