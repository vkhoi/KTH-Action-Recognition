# Get KTH datset.
wget http://www.nada.kth.se/cvap/actions/boxing.zip
wget http://www.nada.kth.se/cvap/actions/handclapping.zip
wget http://www.nada.kth.se/cvap/actions/handwaving.zip
wget http://www.nada.kth.se/cvap/actions/jogging.zip
wget http://www.nada.kth.se/cvap/actions/running.zip
wget http://www.nada.kth.se/cvap/actions/walking.zip

unzip boxing.zip -d boxing
unzip handclapping.zip -d handclapping
unzip handwaving.zip -d handwaving
unzip jogging.zip -d jogging
unzip running.zip -d running
unzip walking.zip -d walking

rm *.zip