wget http://users.cecs.anu.edu.au/~bdm/nauty/nauty26r11.tar.gz

# get tar file and open tar file
wget https://web.cs.dal.ca/~peter/software/pynauty/pynauty-0.6.0.tar.gz
wait

tar xvzf pynauty-0.6.0.tar.gz

wait
cd pynauty-0.6.0
mv ~/nauty26r11.tar.gz ./
tar xvzf nauty26r11.tar.gz

wait 

cd nauty26r11
./configure
wait
make
wait

cd ~/pynauty-0.6.0
ln -s nauty26r11 nauty

cd 
rm pynauty-0.6.0.tar.gz
# nautywrap.c under pynauty-X.Y.Z and graph.py with the nautywrap.c and graph.py
# This assumes your graph.py and nautywrap.c are in the home directory

#wget https://raw.githubusercontent.com/rusty1s/graph-based-image-classification/master/__pynauty__/graph.py
#wget https://raw.githubusercontent.com/rusty1s/graph-based-image-classification/master/__pynauty__/nautywrap.c



wait

cd
mv graph.py ~/pynauty-0.6.0/src/graph.py
mv nautywrap.c ~/pynauty-0.6.0/src/nautywrap.c
wait
cd ~/pynauty-0.6.0
make pynauty


