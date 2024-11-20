use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize, Clone)]
struct Player {
    element: u32,
    name: String,
    value: f32,
    position: String,
    team: String,
    predicted_points: f32,
}

fn read_csv(path: &str) -> Result<Vec<Player>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut players = Vec::new();

    for result in rdr.deserialize() {
        let player: Player = result?;
        players.push(player);
    }

    Ok(players)
}

// Genetic Algorithm parameters
const POPULATION_SIZE: usize = 150;
const GENERATIONS: usize = 2500;
const MUTATION_RATE: f32 = 0.1;
const MAX_VALUE: f32 = 1000.0;
const MAX_PLAYERS_PER_TEAM: usize = 3;

// Constraints for positions
fn max_positions() -> HashMap<String, usize> {
    HashMap::from([
        ("GK".to_string(), 2),
        ("DEF".to_string(), 5),
        ("MID".to_string(), 5),
        ("FWD".to_string(), 3),
    ])
}

// Fitness function
fn fitness(team: &Vec<Player>) -> f32 {
    let total_value: f32 = team.iter().map(|p| p.value).sum();
    if total_value > MAX_VALUE {
        return 0.0;
    }
    let mut points: Vec<_> = team.iter().map(|p| p.predicted_points).collect();

    // Sort using partial_cmp and unwrap to handle floating-point comparisons
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let adjusted_sum: f32 = points
        .iter()
        .enumerate()
        .map(|(i, &pts)| if i < 4 { pts * 0.25 } else { pts })
        .sum();

    adjusted_sum
}

// Check team constraints
fn satisfies_constraints(team: &Vec<Player>) -> bool {
    let mut team_counts = HashMap::new();
    let mut position_counts = HashMap::new();
    let mut total_value = 0.0;

    let mut unique_elements = HashMap::new();
    for player in team {
        if unique_elements.contains_key(&player.element) {
            return false;
        }
        unique_elements.insert(player.element, true);
    }

    for player in team {
        *position_counts.entry(player.position.clone()).or_insert(0) += 1;
        *team_counts.entry(player.team.clone()).or_insert(0) += 1;
        total_value += player.value;

        if let Some(&max_pos) = max_positions().get(&player.position) {
            if position_counts[&player.position] > max_pos {
                return false;
            }
        }
        if team_counts[&player.team] > MAX_PLAYERS_PER_TEAM || total_value > MAX_VALUE {
            return false;
        }
    }
    team.len() == 15
}

// Generate a random team satisfying constraints
fn generate_random_team(players: &Vec<Player>) -> Vec<Player> {
    let mut rng = rand::thread_rng();
    let mut team = Vec::new();
    let mut team_counts = HashMap::new();
    let mut position_counts = HashMap::new();
    let mut total_value = 0.0;

    while team.len() < 15 {
        if let Some(player) = players.choose(&mut rng) {
            let position_count = position_counts.entry(player.position.clone()).or_insert(0);
            let team_count = team_counts.entry(player.team.clone()).or_insert(0);

            if *position_count < *max_positions().get(&player.position).unwrap_or(&0)
                && *team_count < MAX_PLAYERS_PER_TEAM
                && total_value + player.value <= MAX_VALUE
            {
                team.push(player.clone());
                *position_count += 1;
                *team_count += 1;
                total_value += player.value;
            }
        }
    }
    team
}

// Initialize population
fn create_initial_population(players: &Vec<Player>) -> Vec<Vec<Player>> {
    (0..POPULATION_SIZE)
        .map(|_| generate_random_team(players))
        .collect()
}

// Perform crossover while maintaining constraints
fn crossover(parent1: &Vec<Player>, parent2: &Vec<Player>) -> Vec<Player> {
    
    let mut rng = rand::thread_rng();
    let split = rng.gen_range(0..15);
    let mut child = Vec::new();

    child.extend_from_slice(&parent1[..split]);
    child.extend_from_slice(&parent2[split..]);

    // Ensure constraints, remove duplicates, add unique players if necessary
   
    child.dedup_by_key(|p| p.element);
    let mut cur = 0;
    while child.len() < 15 && cur < 400 {
        cur += 1;
        if let Some(player) = parent1.choose(&mut rng) {
            if !child.iter().any(|p| p.element == player.element) {
                child.push(player.clone());
            }
        }
    }

    // Final check to ensure constraints are satisfied
    if satisfies_constraints(&child) {
        child
    } else {
        if rng.gen::<f32>() < 0.5 {
            parent1.clone()
        } else {
            parent2.clone()
        }
    }
}

// Mutation with constraint check
fn mutate(team: &mut Vec<Player>, players: &Vec<Player>) {
    let mut rng = rand::thread_rng();
    if rng.gen::<f32>() < MUTATION_RATE {
        if let Some(index) = (0..team.len()).choose(&mut rng) {
            if let Some(new_player) = players.choose(&mut rng) {
                team[index] = new_player.clone();
            }
        }
    }

    if !satisfies_constraints(team) {
        *team = generate_random_team(players);
    }
}

// Genetic algorithm main function
fn select_best_team_ga(players: Vec<Player>) -> Vec<Player> {
    let mut population = create_initial_population(&players);

    let progress_bar = ProgressBar::new(GENERATIONS as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar().template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        ),
    );

    for g in 0..GENERATIONS {
       
        // Evaluate fitness in parallel
        let fitness_scores: Vec<(f32, Vec<Player>)> = population
            .par_iter()
            .map(|team| (fitness(team), team.clone()))
            .collect();

        // Sort by fitness and select top half
        let mut best_individuals = fitness_scores.clone();
        best_individuals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        best_individuals.truncate(POPULATION_SIZE / 2);

        // Generate new population with crossover and mutation
        let mut new_population = Vec::new();
        while new_population.len() < POPULATION_SIZE {
            let parent1 = &best_individuals.choose(&mut rand::thread_rng()).unwrap().1;
            let parent2 = &best_individuals.choose(&mut rand::thread_rng()).unwrap().1;
            let mut child = crossover(parent1, parent2);
            mutate(&mut child, &players);
            new_population.push(child);
        }

        population = new_population;
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Genetic algorithm complete!");

    // Return the best team from the final population
    population
        .into_iter()
        .max_by(|a, b| fitness(a).partial_cmp(&fitness(b)).unwrap())
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let players = read_csv("./df_encoded_new.csv")?;
    let best_team = select_best_team_ga(players);

    let mut sorted_team = best_team.clone();
    sorted_team.sort_by(|a, b| a.predicted_points.partial_cmp(&b.predicted_points).unwrap());

    for (i, player) in best_team.iter().enumerate() {
        if sorted_team.iter().position(|p| p.element == player.element).unwrap() < 4 {
            println!(
                "Selected: {} - Position: {} - Team: {} - Predicted Points: {} - Value: {} (Bench)",
                player.name, player.position, player.team, player.predicted_points, player.value
            );
        } else {
            println!(
                "Selected: {} - Position: {} - Team: {} - Predicted Points: {} - Value: {}",
                player.name, player.position, player.team, player.predicted_points, player.value
            );
        }
    }

    let (total_sum, max_point) = best_team.iter()
        .map(|p| p.predicted_points)
        .fold((0.0_f32, 0.0_f32), |(sum, max), x| (sum + x, max.max(x)));
    
    let total_predicted_points = total_sum + max_point;

println!("Total Predicted Points (with highest doubled): {}", total_predicted_points);


    

    let total_value: f32 = best_team.iter().map(|p| p.value).sum();
    println!("Total Value: {}", total_value);
    Ok(())
}
