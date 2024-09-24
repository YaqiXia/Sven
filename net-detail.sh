#!/bin/bash
es=${1:-"eno1 eno2"}
ethns=($es)
#RX_pre=(0 0 0 0 0 0 0 0 0 0)
#TX_pre=(0 0 0 0 0 0 0 0 0 0)
log_file=${1:-/dev/null}
if [ ! $log_file == /dev/null ];then
	log_file="$log_file"
	rm $log_file
fi
media=tmp.log

for i in `seq 1 ${#ethns[*]}`
do
  idx=$((i-1))
  ethn=${ethns[idx]}
  cat /proc/net/dev > tmp.log
  RX_pre[$idx]=$(cat tmp.log | grep $ethn | sed 's/:/ /g' | awk '{print $2}')
  TX_pre[$idx]=$(cat tmp.log | grep $ethn | sed 's/:/ /g' | awk '{print $10}')
done
echo `date -R` >> ${log_file}

for i in `seq 1 120`
do
  RX_T=0
  TX_T=0
  s=""
  sleep 1
  clear
  echo -e "\t RX(MB) `date +%k:%M:%S` TX(MB)"
  for i in `seq 1 ${#ethns[*]}`
  do
    idx=$((i-1))
    ethn=${ethns[idx]}
    
    cat /proc/net/dev > tmp.log
    RX_now=$(cat tmp.log | grep $ethn | sed 's/:/ /g' | awk '{print $2}')
    TX_now=$(cat tmp.log | grep $ethn | sed 's/:/ /g' | awk '{print $10}')
    

    RX=$(($RX_now-${RX_pre[idx]}))
    TX=$(($TX_now-${TX_pre[idx]}))
    RX=$(echo $RX | awk '{printf("%.2f", $1/1048576)}')
    TX=$(echo $TX | awk '{printf("%.2f", $1/1048576)}')

    echo -e "$ethn:\t $RX\t $TX"
    s="$s | $ethn: $RX $TX"

    tmp=`echo "$RX_T $RX" |  awk '{printf("%.2f + %.2f",$1,$2)}' `
    RX_T=`echo $tmp | bc`
    tmp=`echo "$TX_T $TX" |  awk '{printf("%.2f + %.2f",$1,$2)}' `
    TX_T=`echo $tmp | bc`
    #echo -e "$ethn \t $RX   $TX" >> ${log_file}
    RX_pre[$idx]=${RX_now}
    TX_pre[$idx]=${TX_now}
  done
  echo -e "total:\t $RX_T \t $TX_T"
  s="[time: `date  +%H:%M:%S`] total: $RX_T $TX_T $s"
  #echo -e "total: $RX_T $TX_T" >> $log_file
  echo -e $s >> $log_file	
done

echo `date -R` >> ${log_file}
