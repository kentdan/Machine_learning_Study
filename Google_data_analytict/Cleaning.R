# Cleaning data
library(pacman)
  p_load(skimr,janitor,tidyverse,palmerpenguins)
skim_without_charts(penguins)
glimpse(penguins)  
penguins %>% 
  select(-species)
#everything except species
rename_with(penguins,tolower)
#rename to lower cases
#clean form janitor package
clean_names(penguins)

#new data
bookings_df <- hotel_booking <- read.csv("~//Downloads/hotel_bookings.csv")
glimpse(bookings_df)
trimmed_df <- bookings_df %>% 
  select(hotel , is_canceled, lead_time)
#trim
trimmed_df %>% 
  select(hotel, is_canceled, lead_time) %>% 
  rename( hotel_type= hotel)
#date unite
example_df <- bookings_df %>%
  select(arrival_date_year, arrival_date_month) %>% 
  unite(arrival_month_year, c("arrival_date_month", "arrival_date_year"), sep = " ")

example_df <- bookings_df %>%
  summarize(number_canceled= sum(is_canceled),average_lead_time= mean(lead_time) )
  
  head(example_df)
#     unite(a,b), sep = " ")
  #sep is seperation
  hotel_summary <- 
    bookings_df %>%
    group_by(hotel) %>%
    summarise(average_lead_time=mean(lead_time),
              min_lead_time=min(lead_time),
              max_lead_time=max(lead_time))
  head(hotel_summary)
  