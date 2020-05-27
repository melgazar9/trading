set -e

echo running 6E...
node 6E/6E_stream.js > ../Data/Futures/6E/6E_1min/6E_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$!
echo "killing hcild pip ${pid}"
@kill -INT ${pid}
sleep 5

echo running CL...
node CL/CL_stream.js > ../Data/Futures/CL/CL_1min/CL_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running ES...  
node ES/ES_stream.js > ../Data/Futures/ES/ES_1min/ES_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running GC... 
node GC/GC_stream.js > ../Data/Futures/GC/GC_1min/GC_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running NG... 
node NG/NG_stream.js > ../Data/Futures/NG/NG_1min/NG_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running VX... 
node VX/VX_stream.js > ../Data/Futures/VX/VX_1min/VX_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running ZB... 
node ZB/ZB_stream.js > ../Data/Futures/ZB/ZB_1min/ZB_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running ZF 
node ZF/ZF_stream.js > ../Data/Futures/ZF/ZF_1min/ZF_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running ZN 
node ZN/ZN_stream.js > ../Data/Futures/ZN/ZN_1min/ZN_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo running ZN 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo running ZT... 
node ZT/ZT_stream.js > ../Data/Futures/ZT/ZT_1min/ZT_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log & sleep 180

pid=$! 
echo "killing hcild pip ${pid}" 
kill -INT ${pid} 
sleep 5 

echo "Done!"
