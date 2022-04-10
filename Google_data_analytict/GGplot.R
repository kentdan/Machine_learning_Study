library(tidyverse)
library(skimr)
library(janitor)
hotel_bookings <- read_csv("https://d3c33hcgiwev3.cloudfront.net/GL0bk8O2Sja9G5PDtko2uQ_31e445d7ca64417eb45aeaa08ec90bf1_hotel_bookings.csv?Expires=1626652800&Signature=lXJrpJ5Mx44STUuGy~04GVW2cDKenB5uds5n9hFBTce2V7w3mmadLVRqxlsVp1rOYS-0zZE1PtwOVTb8ruJr56ruSAtvLh9rd8r0nkW9rsGrV9wWVu25Ju1fV7INxfWeo7g7QvzsD2IGY3BQKZfz50KvlD9XE~TFs93U~c0VPG8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A")
head(hotel_bookings)
glimpse(hotel_bookings)
hotel_city<- filter(hotel_bookings, hotel_bookings$hotel=="City Hotel")
glimpse(hotel_city)
#filter
hotel_summary <- 
  hotel_bookings %>%
  group_by(hotel) %>%
  summarise(average_lead_time=mean(lead_time),
            min_lead_time=min(lead_time),
            max_lead_time=max(lead_time))
head(hotel_summary)
#summary
hotel_bookings_v2 <-
  arrange(hotel_bookings, desc(lead_time))
mean(hotel_bookings_v2$lead_time)
#ggplot

ggplot(data = hotel_bookings) +
  geom_bar(mapping = aes(x = distribution_channel)) +
  facet_wrap(~deposit_type~market_segment) +
  theme(axis.text.x = element_text(angle = 45))
#facet grid
ggplot(data = hotel_bookings) +
  geom_bar(mapping = aes(x = distribution_channel)) +
  facet_grid(~deposit_type) +
  theme(axis.text.x = element_text(angle = 45))
#
ggplot(data = hotel_bookings) +
  geom_bar(mapping = aes(x = distribution_channel)) 
#
ggplot(data = hotel_bookings) +
  geom_point(mapping = aes(x = lead_time, y = children))
#onlineta_city_hotels filter
onlineta_city_hotels <- filter(hotel_bookings, 
                               (hotel=="City Hotel" & 
                                  hotel_bookings$market_segment=="Online TA"))
head(onlineta_city_hotels)
#using pipe
onlineta_city_hotels_v2 <- hotel_bookings %>%
  filter(hotel=="City Hotel") %>%
  filter(market_segment=="Online TA")
#ggplot
ggplot(data = onlineta_city_hotels_v2) +
  geom_point(mapping = aes(x = lead_time, y = children))
#labs
mindate <- min(hotel_bookings$arrival_date_year)
maxdate <- max(hotel_bookings$arrival_date_year)
ggplot(data = hotel_bookings) +
  geom_bar(mapping = aes(x = market_segment)) +
  facet_wrap(~hotel) +
  theme(axis.text.x = element_text(angle = 45)) +
  labs(title="Comparison of market segments by hotel type for hotel bookings",
       caption=paste0("Data from: ", mindate, " to ", maxdate),
       x="Market Segment",
       y="Number of Bookings")
#saving chart
ggsave('hotel_booking_chart.png')
