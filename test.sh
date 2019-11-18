
while getopts "a:" opt
do
    case $opt in
        a) echo "Option a: $opt, argument: $OPTARG";;
    esac
done