import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

class FeatureEngineer:
    """
    Class for extracting and engineering features from the March Madness data.
    This will create features for each team based on their seasonal performance,
    and then combine them into game-level features for model input.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the feature engineer.
        
        Args:
            data_loader: The DataLoader instance with all the data
        """
        self.data_loader = data_loader
        self.mens_data, self.womens_data, self.general_data = data_loader.split_by_gender()
        
        # Cache for team features
        self.team_features = {}
        self.matchup_features = {}
    
    def _get_team_season_stats(self, gender: str, season: int, detailed: bool = True) -> pd.DataFrame:
        """
        Get season statistics for each team.
        
        Args:
            gender: 'M' for men's data, 'W' for women's data
            season: Season year to get stats for
            detailed: Whether to use detailed results (with more stats) if available
            
        Returns:
            DataFrame with team season statistics
        """
        data = self.mens_data if gender == 'M' else self.womens_data
        
        # Choose the appropriate results data
        if detailed and 'RegularSeasonDetailedResults' in data:
            results = data['RegularSeasonDetailedResults']
            has_detailed = True
        else:
            results = data['RegularSeasonCompactResults']
            has_detailed = False
        
        # Filter to the specific season
        season_results = results[results['Season'] == season].copy()
        
        # For winners: Use W* columns for team stats, L* columns for opponent stats
        winners = season_results.copy()
        
        # For winners: team stats are W columns, opponent stats are L columns
        winner_team_cols = {}
        winner_opp_cols = {}
        
        for col in winners.columns:
            if col.startswith('W') and col not in ['WTeamID', 'WScore', 'WLoc']:
                winner_team_cols[col] = 'Team' + col[1:]
            elif col.startswith('L') and col not in ['LTeamID', 'LScore']:
                winner_opp_cols[col] = 'Opponent' + col[1:]
        
        # Special handling for main columns
        winner_team_cols['WTeamID'] = 'TeamID'
        winner_team_cols['WScore'] = 'TeamScore'
        winner_opp_cols['LTeamID'] = 'OpponentID'
        winner_opp_cols['LScore'] = 'OpponentScore'
        
        # Rename all columns
        winners = winners.rename(columns={**winner_team_cols, **winner_opp_cols})
        winners['WonGame'] = 1
                
        # For losers: Use L* columns for team stats, W* columns for opponent stats
        losers = season_results.copy()
        
        # For losers: team stats are L columns, opponent stats are W columns
        loser_team_cols = {}
        loser_opp_cols = {}
        
        for col in losers.columns:
            if col.startswith('L') and col not in ['LTeamID', 'LScore']:
                loser_team_cols[col] = 'Team' + col[1:]
            elif col.startswith('W') and col not in ['WTeamID', 'WScore', 'WLoc']:
                loser_opp_cols[col] = 'Opponent' + col[1:]
        
        # Special handling for main columns
        loser_team_cols['LTeamID'] = 'TeamID'
        loser_team_cols['LScore'] = 'TeamScore'
        loser_opp_cols['WTeamID'] = 'OpponentID'
        loser_opp_cols['WScore'] = 'OpponentScore'
        
        # Rename all columns
        losers = losers.rename(columns={**loser_team_cols, **loser_opp_cols})
        losers['WonGame'] = 0
        
        # Combine winner and loser data
        team_games = pd.concat([winners, losers], ignore_index=True)
        
        # Add derived features
        team_games['ScoreDiff'] = team_games['TeamScore'] - team_games['OpponentScore']
        team_games['ScoreRatio'] = team_games['TeamScore'] / team_games['OpponentScore']
        
        # Calculate additional statistics if detailed results available
        if has_detailed:
            # Calculate percentages only if we have the required columns
            if 'TeamFGM' in team_games.columns and 'TeamFGA' in team_games.columns:
                team_games['TeamFGPercentage'] = team_games['TeamFGM'] / team_games['TeamFGA'].clip(lower=1)
            
            if 'TeamFGM3' in team_games.columns and 'TeamFGA3' in team_games.columns:
                team_games['Team3PPercentage'] = team_games['TeamFGM3'] / team_games['TeamFGA3'].clip(lower=1)
            
            if 'TeamFTM' in team_games.columns and 'TeamFTA' in team_games.columns:
                team_games['TeamFTPercentage'] = team_games['TeamFTM'] / team_games['TeamFTA'].clip(lower=1)
            
            # Calculate efficiency metrics
            if 'TeamOR' in team_games.columns and 'OpponentDR' in team_games.columns:
                denominator = team_games['TeamOR'] + team_games['OpponentDR']
                team_games['TeamOffensiveReboundPercentage'] = team_games['TeamOR'] / denominator.clip(lower=1)
            
            if 'TeamDR' in team_games.columns and 'OpponentOR' in team_games.columns:
                denominator = team_games['TeamDR'] + team_games['OpponentOR']
                team_games['TeamDefensiveReboundPercentage'] = team_games['TeamDR'] / denominator.clip(lower=1)
            
            if all(col in team_games.columns for col in ['TeamTO', 'TeamFGA', 'TeamFTA']):
                denominator = team_games['TeamFGA'] + 0.44 * team_games['TeamFTA'] + team_games['TeamTO']
                team_games['TeamTurnoverPercentage'] = team_games['TeamTO'] / denominator.clip(lower=1)
            
            if 'TeamAst' in team_games.columns and 'TeamFGM' in team_games.columns:
                team_games['TeamAssistPercentage'] = team_games['TeamAst'] / team_games['TeamFGM'].clip(lower=1)
            
            # Calculate Four Factors (effective field goal %, turnover %, offensive rebound %, free throw rate)
            if all(col in team_games.columns for col in ['TeamFGM', 'TeamFGA']):
                if 'TeamFGM3' in team_games.columns:
                    team_games['TeamEffectiveFGPercentage'] = (team_games['TeamFGM'] + 0.5 * team_games['TeamFGM3']) / team_games['TeamFGA'].clip(lower=1)
                else:
                    team_games['TeamEffectiveFGPercentage'] = team_games['TeamFGM'] / team_games['TeamFGA'].clip(lower=1)
            
            if 'TeamFTA' in team_games.columns and 'TeamFGA' in team_games.columns:
                team_games['TeamFreeThrowRate'] = team_games['TeamFTA'] / team_games['TeamFGA'].clip(lower=1)
        
        # Group by team and calculate season averages
        team_stats = team_games.groupby('TeamID').agg({
            'WonGame': 'mean',  # Win percentage
            'TeamScore': 'mean',
            'OpponentScore': 'mean',
            'ScoreDiff': 'mean',
            'ScoreRatio': 'mean',
            # Add more aggregations for detailed stats if available
        })
        
        # Rename columns
        team_stats = team_stats.rename(columns={
            'WonGame': 'WinPercentage', 
            'TeamScore': 'PointsPerGame',
            'OpponentScore': 'OpponentPointsPerGame', 
            'ScoreDiff': 'PointsDifferential',
            'ScoreRatio': 'PointsRatio'
        })
        
        # Add more advanced stats if detailed results are available
        if has_detailed:
            # List of possible detailed stats to aggregate
            possible_detailed_stats = [
                'TeamFGPercentage', 'Team3PPercentage', 'TeamFTPercentage', 
                'TeamOR', 'TeamDR', 'TeamTO', 'TeamAst', 'TeamStl', 'TeamBlk', 'TeamPF',
                'TeamOffensiveReboundPercentage', 'TeamDefensiveReboundPercentage',
                'TeamTurnoverPercentage', 'TeamAssistPercentage',
                'TeamEffectiveFGPercentage', 'TeamFreeThrowRate'
            ]
            
            # Only include columns that actually exist in the data
            detailed_aggs = {}
            for stat in possible_detailed_stats:
                if stat in team_games.columns:
                    detailed_aggs[stat] = 'mean'
            
            # Only proceed if we have some stats to aggregate
            if detailed_aggs:
                detailed_df = team_games.groupby('TeamID').agg(detailed_aggs)
                team_stats = pd.merge(team_stats, detailed_df, on='TeamID')
        
        # Calculate number of games played
        game_counts = team_games.groupby('TeamID').size().to_frame('GamesPlayed')
        team_stats = pd.merge(team_stats, game_counts, on='TeamID')
        
        # Add team strength metrics if available
        if 'MasseyOrdinals' in data:
            ordinals = data['MasseyOrdinals']
            # Filter to end of regular season rankings
            # This assumes there's a way to determine the end of the regular season
            end_season_ordinals = ordinals[(ordinals['Season'] == season) & 
                                           (ordinals['RankingDayNum'] <= 133)].copy()
            
            if not end_season_ordinals.empty:
                # Average the rankings across different systems
                avg_ranks = end_season_ordinals.groupby('TeamID')['OrdinalRank'].mean().to_frame('AvgRank')
                team_stats = pd.merge(team_stats, avg_ranks, on='TeamID', how='left')
        
        # Add team seed as a feature if it's tournament time
        tournament_seeds = data['NCAATourneySeeds']
        season_seeds = tournament_seeds[tournament_seeds['Season'] == season].copy()
        
        if not season_seeds.empty:
            # Extract numeric seed from the seed string (e.g., W01 -> 1)
            season_seeds['SeedValue'] = season_seeds['Seed'].str[1:3].astype(int)
            seed_dict = dict(zip(season_seeds['TeamID'], season_seeds['SeedValue']))
            
            # Add seed values to team stats
            team_stats['Seed'] = team_stats.index.map(seed_dict).fillna(20)  # Default high seed for teams not in tournament
        
        # Make sure the index has a name
        team_stats.index.name = 'TeamID'
        
        return team_stats
    
    def generate_tournament_features(self, gender: str, season: int, stage: int = 2) -> pd.DataFrame:
        """
        Generate features for tournament predictions.
        
        Args:
            gender: 'M' for men's data, 'W' for women's data
            season: Season year to generate features for
            stage: 1 for all possible matchups, 2 for specific bracket matchups
            
        Returns:
            DataFrame with features for each potential tournament matchup
        """
        # Get team stats for the season
        team_stats = self._get_team_season_stats(gender, season)
        
        # If it's a future season, we'll need to create potential matchups
        data = self.mens_data if gender == 'M' else self.womens_data
        
        # Create features for each potential matchup depending on the stage
        if stage == 1:
            # For Stage 1, we create features for all possible matchups of tournament teams
            seeds = data['NCAATourneySeeds']
            season_seeds = seeds[seeds['Season'] == season].copy()
            
            # Get all tournament teams
            teams = sorted(season_seeds['TeamID'].unique())
            
            # Generate all possible matchups
            matchups = []
            for i, team1 in enumerate(teams):
                for team2 in teams[i+1:]:
                    matchups.append({
                        'Season': season,
                        'Team1ID': team1,
                        'Team2ID': team2
                    })
            
            matchups_df = pd.DataFrame(matchups)
            
        else:
            # For Stage 2, we use the tournament slots to determine potential matchups
            slots = data['NCAATourneySlots']
            season_slots = slots[slots['Season'] == season].copy()
            seeds = data['NCAATourneySeeds']
            season_seeds = seeds[seeds['Season'] == season].copy()
            
            # TODO: Generate matchups based on bracket structure
            # This is a placeholder for the actual tournament matchup generation
            matchups_df = pd.DataFrame({
                'Season': [season],
                'Team1ID': [season_seeds['TeamID'].iloc[0]],
                'Team2ID': [season_seeds['TeamID'].iloc[1]]
            })
        
        # Create matchup features
        matchup_features = self._create_matchup_features(matchups_df, team_stats)
        
        return matchup_features
    
    def _create_matchup_features(self, matchups_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create matchup-level features from team statistics.
        
        Args:
            matchups_df: DataFrame with Season, Team1ID, Team2ID columns
            team_stats: DataFrame with team-level statistics
            
        Returns:
            DataFrame with features for each matchup
        """
        # Initialize the result DataFrame
        result = matchups_df.copy()
        
        # Ensure the team_stats index has a name
        if team_stats.index.name is None:
            team_stats.index.name = 'TeamID'
        
        # Add team1 features - reset_index to convert the index to a column
        team1_stats = team_stats.reset_index()
        team1_stats = team1_stats.rename(
            columns={col: f'Team1_{col}' for col in team1_stats.columns if col != 'TeamID'}
        )
        team1_stats = team1_stats.rename(columns={'TeamID': 'Team1ID'})
        result = pd.merge(result, team1_stats, on='Team1ID', how='left')
        
        # Add team2 features - reset_index to convert the index to a column
        team2_stats = team_stats.reset_index()
        team2_stats = team2_stats.rename(
            columns={col: f'Team2_{col}' for col in team2_stats.columns if col != 'TeamID'}
        )
        team2_stats = team2_stats.rename(columns={'TeamID': 'Team2ID'})
        result = pd.merge(result, team2_stats, on='Team2ID', how='left')
        
        # Create comparative features (differences and ratios)
        for col in team_stats.columns:
            # Only create comparative features for numeric columns
            if pd.api.types.is_numeric_dtype(team_stats[col]):
                result[f'Diff_{col}'] = result[f'Team1_{col}'] - result[f'Team2_{col}']
                
                # For ratio features, avoid division by zero
                # We'll only do this for strictly positive values
                if (team_stats[col] > 0).all():
                    result[f'Ratio_{col}'] = result[f'Team1_{col}'] / result[f'Team2_{col}']
        
        # Calculate seed difference
        if 'Team1_Seed' in result.columns and 'Team2_Seed' in result.columns:
            result['SeedDiff'] = result['Team1_Seed'] - result['Team2_Seed']
            result['HigherSeed'] = (result['Team1_Seed'] < result['Team2_Seed']).astype(int)
        
        return result
    
    def prepare_historical_training_data(self, gender: str, start_season: int, end_season: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare historical data for model training.
        
        Args:
            gender: 'M' for men's data, 'W' for women's data
            start_season: First season to include
            end_season: Last season to include
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        data = self.mens_data if gender == 'M' else self.womens_data
        
        # Get tournament results
        tourney_results = data['NCAATourneyCompactResults']
        
        # Filter to the desired seasons
        season_filter = (tourney_results['Season'] >= start_season) & (tourney_results['Season'] <= end_season)
        filtered_results = tourney_results[season_filter].copy()
        
        # Create a DataFrame of matchups from the results
        matchups = []
        for _, row in filtered_results.iterrows():
            # For each game, create two matchups - one with the actual outcome and one with teams swapped
            # First matchup: Team1 = winner, Team2 = loser (Team1 wins)
            matchups.append({
                'Season': row['Season'],
                'Team1ID': row['WTeamID'],
                'Team2ID': row['LTeamID'],
                'Team1Won': 1  # Team1 is the winner
            })
            
            # Second matchup: Team1 = loser, Team2 = winner (Team1 loses)
            matchups.append({
                'Season': row['Season'],
                'Team1ID': row['LTeamID'],
                'Team2ID': row['WTeamID'],
                'Team1Won': 0  # Team1 is the loser
            })
        
        matchups_df = pd.DataFrame(matchups)
        
        # Generate features for each season
        all_features = []
        
        for season in range(start_season, end_season + 1):
            # Get team stats for the season
            team_stats = self._get_team_season_stats(gender, season)
            
            # Filter matchups to the current season
            season_matchups = matchups_df[matchups_df['Season'] == season].copy()
            
            # Create matchup features
            if not season_matchups.empty:
                matchup_features = self._create_matchup_features(season_matchups, team_stats)
                all_features.append(matchup_features)
        
        # Combine features from all seasons
        X = pd.concat(all_features, ignore_index=True)
        
        # Extract the target variable
        y = X['Team1Won'].values
        
        # Drop columns that shouldn't be used as features, but keep Season for splitting
        columns_to_drop = ['Team1ID', 'Team2ID', 'Team1Won']
        
        # Keep a copy of X with Season for splitting
        X_with_season = X.drop(columns=columns_to_drop)
        
        # Handle missing values
        X_with_season = X_with_season.fillna(0)
        
        return X_with_season, y
    
    def split_train_validation_test(self, X: pd.DataFrame, y: np.ndarray, 
                                     val_seasons: List[int] = None, 
                                     test_seasons: List[int] = None,
                                     random_split: bool = False,
                                     test_size: float = 0.2,
                                     val_size: float = 0.2,
                                     random_state: int = 42) -> Tuple:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            val_seasons: List of seasons to use for validation (if not random split)
            test_seasons: List of seasons to use for testing (if not random split)
            random_split: Whether to use random splitting instead of by season
            test_size: Proportion of data to use for testing if using random split
            val_size: Proportion of data to use for validation if using random split
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if random_split:
            from sklearn.model_selection import train_test_split
            
            # First split off the test set
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Then split the training set into training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=val_size / (1 - test_size),  # Adjust validation size
                random_state=random_state,
                stratify=y_train_val
            )
            
        else:
            # Split by seasons
            # Assumption: The 'Season' column is still in X
            if 'Season' not in X.columns:
                raise ValueError("Season column is required for splitting by season")
            
            # Default to using the most recent seasons for validation and testing
            if val_seasons is None or test_seasons is None:
                all_seasons = sorted(X['Season'].unique())
                if val_seasons is None:
                    val_seasons = [all_seasons[-2]]  # Second-to-last season
                if test_seasons is None:
                    test_seasons = [all_seasons[-1]]  # Last season
            
            # Create masks for each split
            test_mask = X['Season'].isin(test_seasons)
            val_mask = X['Season'].isin(val_seasons)
            train_mask = ~(test_mask | val_mask)
            
            # Split the data
            X_train = X[train_mask].drop(columns=['Season'])
            X_val = X[val_mask].drop(columns=['Season'])
            X_test = X[test_mask].drop(columns=['Season'])
            
            y_train = y[train_mask]
            y_val = y[val_mask]
            y_test = y[test_mask]
        
        return X_train, X_val, X_test, y_train, y_val, y_test 