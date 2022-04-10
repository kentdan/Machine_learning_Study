library(tidyverse)
#vectors
typeof(c(1L , 3L))
c(2.5, 48.5, 101.5)
c(TRUE, FALSE, TRUE)
x <- c(33.5, 57.75, 120.05)
length(x)
y <- c(TRUE, TRUE, FALSE)
is.character(y)
typeof(y)
#Explicit coercion as.logical(), as.integer(), as.double(), or as.character()
#list
a <- (list("a", 1L, 1.5, TRUE))
str(a)
z <- list(list(list(1 , 3, 5)))
# $ symbols reflect the nested structure of this list. 
xa <- sample(20, 100, replace = TRUE)
xa
ya <- xa > 10
sum(ya)  # how many are greater than 10?
mean(ya) # what proportion are greater than 10?
sample(10) + 100
#date
library(lubridate)
today()
now()
#year month date
ymd("2021-01-20")
as_date(now())
#data frame
data.frame(x = c(1, 2, 3) , y = c(1.5, 5.5, 7.5))
#file
dir.create ("destination_folder")
#file.create (“new_text_file.txt”) 
#file.create (“new_csv_file.csv”)
#file.copy (“new_text_file.txt” , “destination_folder”)
#matrix
matrix(c(3:8), nrow = 2)
matrix(c(3:8), ncol = 2)

#ggplot
head(diamonds)
ggplot(data = diamonds, aes(x = carat, y = price, color = cut)) +
  geom_point() +
  facet_wrap(~cut)
#facet_wrap() is an R function used to create subplots
glimpse(diamonds)
browseVignettes("tidyverse") 
#pipe
data("ToothGrowth")
filltered_tg <- filter(ToothGrowth,dose==0.5)
head(filltered_tg)
Filltered_tooth <- ToothGrowth %>% 
  filter(dose==0.5) %>% 
  arrange(len)
head(Filltered_tooth)

Filltered_tooth_sum <- ToothGrowth %>% 
  filter(dose==0.5) %>% 
  group_by(supp) %>% 
  summarize(mean_len = mean(len,na.rm = T),group="drop")
head(Filltered_tooth_sum)
#dataframe
mutate(diamonds,carat_2=carat*100)
str(diamonds)
glimpse(diamonds)
as_tibble(diamonds)
# read_excel
bookings_df <- hotel_booking <- read.csv("~//Downloads/hotel_bookings.csv")
glimpse(hotel_booking)
new_df <- select(bookings_df, `adr`, adults)
mutate(new_df, total = `adr` / adults)

penguins %>%
  arrange(bill_length_mm)