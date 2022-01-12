# Name- Deeksha RV
# Country - India
# Date of Submission - 12 January,2022 
# Project Goal - Creating a Movie Recommendation System

#Import the following libraries:
library(tidyverse)
library(caret)
library(data.table)
library(caret)
library(stringr)
library(knitr)
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Retain only edx and validation datasets
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# View edx dataset for analysis
view(edx)

# Find relationships between various variables
# How is rating spread across the dataset? 
edx %>% ggplot(aes(rating)) + geom_histogram()
# Observation: 4 is the most popular rating given to movies in edx dataset
#              Whole star ratings are more than half star ratings

# Which movies have a 4 star rating?
edx %>% filter(rating == 4) %>% summarise(genres)

# What unique genres exist ?
str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>%
  unique()

# Genres
# How is each genre rated ?
unique_genres <- c("Comedy", "Romance", "Action","Crime","Thriller","Drama","Sci-Fi", "Adventure","Children","Fantasy","War","Animation","Musical","Western","Mystery","Film-Noir","Horror","Documentary","IMAX","(no genres listed)")
sapply(unique_genres, function(g) {
       sum(str_detect(edx$genres, g))
     })

# Drama and Comedy have the highest number of ratings in edx dataset


# Take all movies having 5 star ratings and analyse them 
# Create dataset containing only 5 star movies
Five_star_movies <- edx %>% filter(rating == 5)
#Number of movies rated 5 stars is 1,390,114

# Find genres with highest number of ratings within 5 star movies:
sapply(unique_genres, function(g) {
       sum(str_detect(Five_star_movies$genres, g))
     })
# Drama and Comedy have the highest number of movie ratings among 5 star movies

# Overall genre analysis indicates that Drama and Comedy are the most popular genres of the total edx dataset

# Users
# How many distinct users are there ?
n_distinct(edx$userId)

# How many movies did each user give 5 stars to ?
user_5stars <- edx %>% group_by(userId) %>% summarise(sum(rating == 5))
# Obervation: while some users rated all watched movies 5 stars, some hardly rated 5 stars

## ____ Building the algorithm _____

# First split edx data into test and train sets:
edx_test_index <- createDataPartition(edx$rating, times = 1, p = 0.5, list = FALSE)
edx_test_set <- edx[edx_test_index, ]
edx_train_set <- edx[-edx_test_index, ]


# In order to build the algorithm, we need to know the parameters most suitable to make accurate predictions.
# Typical recommendation system predicts different ratings based on user input
# Start with a simple model that gives same rating for all movies regardless of user
# Model is given in the form of a regression equation, taking Y as dependant var, X as independent var, beta (b) as parameter and eps (epsilon) as error
# Y = f(X,b) + eps
# Simplify this equation into:
# Y = mu + eps where Y is predicted rating and mu is actual rating, given by average of all ratings in edx dataset
mu <- mean(edx_train_set$rating)
# Comes to 3.512 approximately

# To evaluate the model based on mu, calculate RMSE:
rmse1 <- RMSE(edx_train_set$rating, mu)
rmse1
# Comes to 1.060247
# Including more parameters lowers our RMSE and hence improves our model

# Create results table 
options(pillar.sigfig = 7)
results_table <- tibble(Model = "Average rating only", RMSE = rmse1)
results_table

# The movies parameter
# Check how including movie-effect improves algorithm 
movie_effect <- edx_train_set %>% 
                group_by(movieId) %>%
               summarise(b_m = mean(rating - mu))

pred_movie <- mu + edx_test_set %>%
         left_join(movie_effect, by = "movieId") %>%
         pull(b_m)
         
# evaluate RMSE
rmse2 <- RMSE(pred_movie, edx_test_set$rating, na.rm = TRUE)
rmse2
#RMSE comes to 0.9441335, much lesser than rmse1.

results_table <- bind_rows(results_table, tibble(Model = "Movie model", RMSE = rmse2))
results_table

# The user parameter
user_effect <- edx_train_set %>%
                left_join(movie_effect, by = "movieId") %>%
                group_by(userId) %>%
                summarise(b_u = mean(rating - mu - b_m))

pred_user <- edx_test_set %>%
              left_join(movie_effect, by = "movieId") %>%
              left_join(user_effect, by = "userId") %>%
              mutate(value = mu + b_m + b_u) %>%
              pull(value)
              
# evaluate rmse
rmse3 <- RMSE(pred_user, edx_test_set$rating, na.rm = TRUE)
rmse3

# Rmse has decreased to 0.8696; further improvement of model.

results_table <- results_table %>% bind_rows(tibble(Model = "Movie + User", RMSE = rmse3))
results_table

#genres parameter
genres_effect <- edx_train_set %>%
                left_join(movie_effect, by = "movieId") %>%
                left_join(user_effect, by = "userId") %>%
                group_by(genres) %>%
                summarise(b_g = mean(rating - mu - b_m - b_u))

pred_genres <- edx_test_set %>%
               left_join(movie_effect, by = "movieId") %>%
               left_join(user_effect, by = "userId") %>%
               left_join(genres_effect, by = "genres") %>%
               mutate(value = mu + b_m + b_u + b_g) %>%
               pull(value)
  
rmse4 <- RMSE(pred_genres, edx_test_set$rating, na.rm = TRUE)
rmse4

#rmse is lowered to 0.86931

results_table <- results_table %>% bind_rows(tibble(Model = "Movie + User + Genres", RMSE = rmse4))
results_table

#regularization
lambdas <- seq(0,10,0.20)

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train_set$rating)
  
  b_m <- edx_train_set %>%
    group_by(movieId) %>%
    summarise(b_m = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train_set %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_m)/(n()+l))
  
  b_g <- edx_train_set %>%
        left_join(b_m, by="movieId") %>%
        left_join(b_u, by ="userId") %>%
        group_by(genres) %>%
        summarise(b_g = sum(rating - mu - b_m - b_u)/(n()+l))
  
  predicted_ratings <- edx_test_set %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by =  "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(reg_pred = mu + b_m + b_u + b_g) %>%
    pull(reg_pred)
  return(RMSE(predicted_ratings, edx_test_set$rating, na.rm = TRUE))
})

rmse5 <- min(rmses)
rmse5

results_table <- results_table %>% bind_rows(tibble(Model = "Regularized model", RMSE = rmse5))
results_table

# optimized lambda value 
lambda <- lambdas[which.min(rmses)]

# this comes to 4.75

# plot lambdas vs rmses
qplot(rmses, lambdas)

# use lambda to improve model parameters and take whole edx data:
mu_edx <- mean(edx$rating)

b_m_edx <- edx %>%
           group_by(movieId) %>%
           summarise(b_m = sum(rating - mu_edx)/(n()+ lambda))

b_u_edx <- edx %>%
           left_join(b_m_edx, by = "movieId") %>%
           group_by(userId) %>%
           summarise(b_u = sum(rating - mu_edx - b_m)/(n() + lambda))

b_g_edx <- edx %>%
           left_join(b_m_edx, by = "movieId") %>%
           left_join(b_u_edx, by = "userId") %>%
           group_by(genres) %>%
           summarise(b_g = sum(rating - mu_edx - b_m - b_u)/(n()+lambda))

# test final model on validation dataset
pred_validation <- validation %>%
                  left_join(b_m_edx, by = "movieId") %>%
                  left_join(b_u_edx, by = "userId") %>%
                  left_join(b_g_edx, by = "genres") %>%
                  mutate(value = mu_edx + b_m + b_u + b_g) %>%
                  pull(value)

# final rmse
final_rmse <- RMSE(pred_validation, validation$rating)
# comes to 0.8644514 - final rmse for this model

options(pillar.sigfig = 7)
results_table <- results_table %>% bind_rows(tibble(Model = "Final rmse", RMSE = final_rmse))
results_table
