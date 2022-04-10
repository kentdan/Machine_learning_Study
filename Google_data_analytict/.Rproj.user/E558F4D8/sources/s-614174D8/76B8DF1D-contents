install.packages('Tmisc')
install.packages('datasauRus')
library(Tmisc)
library(datasauRus)
data(quartet)
glimpse(quartet)
quartet %>% 
  group_by(set) %>% 
  summarize(mean(x),sd(x),cor(x,y))
#plot each set
ggplot(quartet,aes(x,y)) + geom_point() + geom_smooth(method = lm,se = F)+
  facet_wrap(~set)
#datasaurus
datasaurus_dozen %>%
  ggplot(aes(x, y, color = dataset)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~dataset, ncol = 5)
