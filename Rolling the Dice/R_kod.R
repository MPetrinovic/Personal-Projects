## ploting the game:

install.packages("randomcoloR")                               
library("randomcoloR")
set.seed(1983765)                                             
palette4 = distinctColorPalette(10)                    
palette4  

all_walks = c()
for (i in 1:10) {
  random_walk=c(0)
  for (x in 1:150) {
    step = random_walk[length(random_walk)]
    dice = sample(1:6, 1, replace=TRUE)
    if(dice <= 2){step = max(0,step-1)}
    if(dice <= 5) {step = step + 1}
    if(dice == 6) {step = step + 0}
  random_walk = append(random_walk, step)
  }
all_walks = rbind(all_walks, random_walk)
}

all_walks = t(all_walks)
all_walks = ts(all_walks)

ts.plot(all_walks, col= palette4, xlab = 'Throws', ylab='Steps')

## simulate 500 repeats

all_walks = c()
for (i in 1:500) {
  random_walk=c(0)
  for (x in 1:150) {
    step = random_walk[length(random_walk)]
    dice = sample(1:6, 1, replace=TRUE)
    if(dice <= 2){step = max(0,step-1)}
    if(dice <= 5) {step = step + 1}
    if(dice == 6) {step = step + 0}
  random_walk = append(random_walk, step)
  }
all_walks = rbind(all_walks, random_walk)
}

all_walks = t(all_walks)
all_walks = ts(all_walks)

ends = all_walks[151,]
hist(ends, col = 'lightgray')
(probs = length(ends[ends>=75])/500)
