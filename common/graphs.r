library(gdata)
library(plyr)
library(ggplot2)

# when k varies
x = 2:14
accuracy = c(0.923000, 0.928000, 0.920000, 0.924000, 0.914000, 0.913000, 0.909000, 0.914000, 0.913000, 0.915000, 0.902000, 0.906000, 0.901000)
# when split varies
x = 1:9 * 0.1
accuracy = c(0.824889, 0.873250, 0.894857, 0.907000, 0.908000, 0.917000, 0.926667, 0.918000, 0.928000)

data = data.frame(Split=x, Accuracy=accuracy)
ggplot(data, aes(x=Split, y=Accuracy)) + geom_line() + ylim(0.80, 1)
