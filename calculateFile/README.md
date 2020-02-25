1.投票统计结果在 cVoting.txt gVoting(sort).txt 中。
  行属性为(i:参考点id; idy_max:与参考点对应的model点id; MaxAccum:idy_max 所得票数)

2.存在一个小问题：  i:870 点在CPU计算中获得票数为253票 而在GPU计算中获得票数为254票(但是当前投票结果都为 model 第44个点，猜想:与kernel计算中的float 进位，取舍有关)。
