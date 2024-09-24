#!/bin/bash

# by process name
p=${1:-"python"}
kill -9 `ps -ef | grep $p | awk '{ print $2 }'`

# by socket port
# p=${1:-"8888"}
# kill -9 `netstat -nlp | grep $p | awk '{ print $2 }'`
