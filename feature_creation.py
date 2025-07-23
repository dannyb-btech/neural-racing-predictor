#!/usr/bin/env python3
"""
Fixed Bayesian Training Data Extractor

This version fixes the None comparison issues in date handling.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RaceTarget:
    """Target race to analyze"""
    date: str  # "2025-07-19"
    venue: str  # "Flemington" 
    race_number: int  # 5

@dataclass
class HistoricalRaceRecord:
    """Individual historical race performance record for training"""
    # Target information (what we're predicting for)
    target_race_date: str
    target_venue: str
    target_race_number: int
    target_distance: int
    target_track_condition: str
    target_class: str
    
    # Horse identification
    horse_name: str
    horse_code: str
    
    # Historical race details
    race_date: str
    venue: str
    distance: int
    track_condition: str
    track_rating: Optional[float]
    race_class: str
    field_size: int
    
    # Performance outcomes (what we're training to predict)
    finish_position: int
    won: bool  # 1st place
    placed: bool  # 1st, 2nd, or 3rd
    
    # Race context features
    barrier: int
    weight: float
    jockey: str
    trainer: str
    handicap_rating: Optional[float]
    
    # Timing performance (key predictive features)
    standard_time_diff_overall: Optional[float]
    standard_time_diff_800m: Optional[float]
    standard_time_diff_400m: Optional[float] 
    standard_time_diff_final: Optional[float]
    sectional_800m: Optional[float]
    sectional_400m: Optional[float]
    final_sectional: Optional[float]
    finish_time: Optional[float]
    
    # Derived features for modeling
    days_to_target_race: int
    distance_difference: int
    same_venue: bool
    same_track_condition: bool
    same_distance: bool
    same_class: bool
    
    # Career context at time of this race
    career_starts_to_date: int
    career_wins_to_date: int
    career_win_rate_to_date: float
    recent_form_3_races: Optional[float]
    
    # Performance context at time of this race (venue/distance/condition specific)
    venue_win_rate: float
    venue_place_rate: float
    venue_avg_finish: float
    venue_experience: float
    venue_recent_form: float
    distance_win_rate: float
    distance_place_rate: float
    distance_avg_finish: float
    distance_experience: float
    distance_recent_form: float
    condition_win_rate: float
    condition_place_rate: float
    condition_avg_finish: float
    condition_experience: float
    condition_recent_form: float

class BayesianTrainingDataExtractor:
    """Extract training data for Bayesian horse racing models with proper error handling"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        
    def _safe_int(self, value, default=0):
        """Safely convert value to int with fallback."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                value = value.replace('m', '').replace('metres', '').strip()
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value, default=None):
        """Safely convert value to float with fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_date_parse(self, date_str):
        """Safely parse date string with proper None handling."""
        if not date_str or date_str is None:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    
    def find_race(self, target: RaceTarget) -> Tuple[str, int]:
        """Find the specific race and return meet_code and race_number."""
        try:
            target_date = datetime.strptime(target.date, "%Y-%m-%d")
            year = target_date.year
            month = target_date.month
            
            logger.info(f"🔍 Searching for {target.date} {target.venue} Race {target.race_number}")
            
            meetings = self.api_client.get_meetings_by_month(year, month)
            
            target_meetings = []
            for meeting in meetings:
                meeting_date = meeting.get('date', '')
                venue_name = meeting.get('venue', '')
                
                if (meeting_date.startswith(target.date) and 
                    target.venue.lower() in venue_name.lower()):
                    target_meetings.append(meeting)
            
            if not target_meetings:
                raise ValueError(f"No meetings found for {target.date} at {target.venue}")
            
            for meeting in target_meetings:
                meet_code = meeting.get('id') or meeting.get('meetCode')
                if not meet_code:
                    continue
                    
                try:
                    races = self.api_client.get_races_for_meet(meet_code)
                    
                    for race in races:
                        if race.get('raceNumber') == target.race_number:
                            logger.info(f"✅ Found race: {meet_code} Race {target.race_number}")
                            return meet_code, target.race_number
                            
                except Exception as e:
                    logger.warning(f"Error checking meeting {meet_code}: {e}")
                    continue
            
            raise ValueError(f"Race {target.race_number} not found at {target.venue} on {target.date}")
            
        except Exception as e:
            logger.error(f"Error finding race: {e}")
            raise
    
    def extract_target_race_horses(self, meet_code: str, race_number: int) -> List[Dict]:
        """Extract all horses from the target race."""
        try:
            logger.info(f"🐎 Extracting horses from target race {race_number}")
            
            race_data = self.api_client.get_race_entries(meet_code, race_number)
            
            horses = []
            form_entries = race_data.get('formRaceEntries', [])
            
            for entry in form_entries:
                if entry.get('scratched', False):
                    continue
                    
                horse_info = {
                    'horse_name': entry.get('horseName'),
                    'horse_code': entry.get('horseCode'),
                    'barrier': entry.get('barrierNumber'),
                    'weight': entry.get('weight'),
                    'jockey': entry.get('jockeyName'),
                    'trainer': entry.get('trainerName'),
                    'race_data': race_data
                }
                horses.append(horse_info)
            
            logger.info(f"Found {len(horses)} active horses in target race")
            return horses
            
        except Exception as e:
            logger.error(f"Error extracting target race horses: {e}")
            raise
    
    def extract_training_records(self, target: RaceTarget) -> List[HistoricalRaceRecord]:
        """Extract all historical race records for horses in the target race."""
        try:
            logger.info(f"🎯 Extracting training data for {target.date} {target.venue} Race {target.race_number}")
            
            meet_code, race_number = self.find_race(target)
            horses = self.extract_target_race_horses(meet_code, race_number)
            
            if not horses:
                raise ValueError("No horses found in target race")
            
            race_data = horses[0]['race_data']
            target_distance = self._safe_int(race_data.get('distance'), 1200)
            target_track_condition = race_data.get('trackCondition', 'Good')
            target_class = race_data.get('class', '')
            
            logger.info(f"Target race: {target_distance}m, {target_track_condition} track, {target_class}")
            
            all_training_records = []
            target_race_date = datetime.strptime(target.date, "%Y-%m-%d")
            
            for i, horse_info in enumerate(horses, 1):
                horse_name = horse_info['horse_name']
                horse_code = horse_info['horse_code']
                
                logger.info(f"Processing horse {i}/{len(horses)}: {horse_name}")
                
                try:
                    historical_races = self.api_client.get_horse_form(horse_code, max_races=200)
                    
                    real_races = []
                    for race in historical_races:
                        if race is None:
                            continue
                        race_info = race.get('race', {})
                        if race_info is None:
                            # Skip races with no race info rather than create synthetic data
                            continue
                        if not race_info.get('isTrial', False) and not race_info.get('isJumpOut', False):
                            real_races.append(race)
                    
                    horse_records = self._convert_horse_history_to_records(
                        horse_info, real_races, target, target_race_date,
                        target_distance, target_track_condition, target_class
                    )
                    
                    all_training_records.extend(horse_records)
                    
                    logger.info(f"✅ {horse_name}: Added {len(horse_records)} historical race records")
                    
                except Exception as e:
                    import traceback
                    logger.error(f"❌ Failed to process {horse_name}: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
            
            logger.info(f"🏁 Extracted {len(all_training_records)} total training records from {len(horses)} horses")
            
            # Debug info about training data distribution
            if all_training_records:
                horse_counts = {}
                for record in all_training_records:
                    horse_name = record.horse_name
                    if horse_name not in horse_counts:
                        horse_counts[horse_name] = 0
                    horse_counts[horse_name] += 1
                
                logger.info(f"Training records per horse: {dict(sorted(horse_counts.items(), key=lambda x: x[1], reverse=True))}")
            
            return all_training_records
            
        except Exception as e:
            logger.error(f"Error extracting training records: {e}")
            raise
    
    def _convert_horse_history_to_records(self, horse_info: Dict, historical_races: List[Dict], 
                                        target: RaceTarget, target_race_date: datetime,
                                        target_distance: int, target_track_condition: str, 
                                        target_class: str) -> List[HistoricalRaceRecord]:
        """Convert a horse's historical races into training records with proper error handling."""
        
        records = []
        horse_name = horse_info['horse_name']
        horse_code = horse_info['horse_code']
        
        # Sort races by date (most recent first) with safe comparison
        def safe_sort_key(race_item):
            """Safe sorting key that handles None dates"""
            try:
                if race_item is None:
                    return '1900-01-01'
                race_info = race_item.get('race', {})
                date_str = race_info.get('date', '')
                if not date_str:
                    return '1900-01-01'  # Very old date for None values
                return date_str
            except:
                return '1900-01-01'
        
        sorted_races = sorted(historical_races, key=safe_sort_key, reverse=True)
        
        for race_idx, race in enumerate(sorted_races):
            try:
                if race is None:
                    continue
                race_info = race.get('race', {})
                if race_info is None:
                    continue
                
                # Parse race date with proper None handling and debugging
                race_date_str = race_info.get('date')
                if not race_date_str:
                    logger.debug(f"Skipping race {race_idx}: no date")
                    continue
                    
                race_date = self._safe_date_parse(race_date_str)
                if race_date is None:
                    logger.debug(f"Skipping race {race_idx}: invalid date {race_date_str}")
                    continue
                
                # Debug the date comparison
                logger.debug(f"Comparing race_date {race_date} with target_race_date {target_race_date}")
                
                # Only include races before target race
                if race_date >= target_race_date:
                    logger.debug(f"Skipping race {race_idx}: after target date")
                    continue
                
                # Parse finish position
                finish = race.get('finish')
                finish_position = self._parse_finish_position(finish)
                if finish_position is None:
                    continue
                
                # Extract basic race details with safe conversions
                venue_info = race_info.get('venue', {})
                if venue_info is None:
                    venue = 'Unknown'
                elif isinstance(venue_info, dict):
                    venue = venue_info.get('venueName', 'Unknown')
                else:
                    venue = str(venue_info)
                
                distance = self._safe_int(race_info.get('distance'), 1200)
                track_condition = race_info.get('trackCondition', 'Good')
                track_rating = self._safe_float(race_info.get('trackRating'))
                race_class = race_info.get('class', '')
                field_size = self._safe_int(race_info.get('runnersCount'), 8)
                
                # Performance outcomes
                won = (finish_position == 1)
                placed = (finish_position <= 3)
                
                # Race context with safe conversions
                barrier = self._safe_int(race.get('barrierNumber'), 8)
                weight = self._safe_float(race.get('weightCarried'), 57.0)
                jockey = race.get('jockeyName', 'Unknown')
                trainer = race.get('trainerName', 'Unknown')
                handicap_rating = self._safe_float(race.get('handicapRating'))
                
                # Extract timing data
                timing_data = self._extract_timing_features(race)
                
                # Calculate derived features
                days_to_target = (target_race_date - race_date).days
                distance_difference = target_distance - distance
                same_venue = (target.venue and venue and target.venue.lower() in venue.lower()) if target.venue and venue else False
                same_track_condition = (target_track_condition == track_condition)
                same_distance = (target_distance == distance)
                same_class = (target_class and race_class and target_class.lower() in race_class.lower()) if target_class and race_class else False
                
                # Calculate career context at time of this race
                career_context = self._calculate_career_context_at_race(
                    sorted_races, race_idx, race_date
                )
                
                # Calculate performance context at time of this race (using only prior races)
                performance_context = self._calculate_performance_context_at_race(
                    sorted_races, race_idx, race_date, target.venue, target_distance, target_track_condition
                )
                
                # Create training record
                record = HistoricalRaceRecord(
                    target_race_date=target.date,
                    target_venue=target.venue,
                    target_race_number=target.race_number,
                    target_distance=target_distance,
                    target_track_condition=target_track_condition,
                    target_class=target_class,
                    
                    horse_name=horse_name,
                    horse_code=horse_code,
                    
                    race_date=race_date_str,
                    venue=venue,
                    distance=distance,
                    track_condition=track_condition,
                    track_rating=track_rating,
                    race_class=race_class,
                    field_size=field_size,
                    
                    finish_position=finish_position,
                    won=won,
                    placed=placed,
                    
                    barrier=barrier,
                    weight=weight,
                    jockey=jockey,
                    trainer=trainer,
                    handicap_rating=handicap_rating,
                    
                    standard_time_diff_overall=timing_data['overall'],
                    standard_time_diff_800m=timing_data['800m'],
                    standard_time_diff_400m=timing_data['400m'],
                    standard_time_diff_final=timing_data['final'],
                    sectional_800m=timing_data['sect_800m'],
                    sectional_400m=timing_data['sect_400m'],
                    final_sectional=timing_data['sect_final'],
                    finish_time=timing_data['finish_time'],
                    
                    days_to_target_race=days_to_target,
                    distance_difference=distance_difference,
                    same_venue=same_venue,
                    same_track_condition=same_track_condition,
                    same_distance=same_distance,
                    same_class=same_class,
                    
                    career_starts_to_date=career_context['starts'],
                    career_wins_to_date=career_context['wins'],
                    career_win_rate_to_date=career_context['win_rate'],
                    recent_form_3_races=career_context['recent_form'],
                    
                    # Performance context features
                    venue_win_rate=performance_context['venue_win_rate'],
                    venue_place_rate=performance_context['venue_place_rate'],
                    venue_avg_finish=performance_context['venue_avg_finish'],
                    venue_experience=performance_context['venue_experience'],
                    venue_recent_form=performance_context['venue_recent_form'],
                    distance_win_rate=performance_context['distance_win_rate'],
                    distance_place_rate=performance_context['distance_place_rate'],
                    distance_avg_finish=performance_context['distance_avg_finish'],
                    distance_experience=performance_context['distance_experience'],
                    distance_recent_form=performance_context['distance_recent_form'],
                    condition_win_rate=performance_context['condition_win_rate'],
                    condition_place_rate=performance_context['condition_place_rate'],
                    condition_avg_finish=performance_context['condition_avg_finish'],
                    condition_experience=performance_context['condition_experience'],
                    condition_recent_form=performance_context['condition_recent_form']
                )
                
                records.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing race for {horse_name}: {e}")
                continue
        
        return records
    
    def _parse_finish_position(self, finish) -> Optional[int]:
        """Parse finish position handling various formats."""
        if not finish:
            return None
        try:
            if isinstance(finish, str):
                finish_num = int(finish.replace('st', '').replace('nd', '').replace('rd', '').replace('th', ''))
            else:
                finish_num = int(finish)
            return finish_num
        except (ValueError, TypeError):
            return None
    
    def _extract_timing_features(self, race: Dict) -> Dict[str, Optional[float]]:
        """Extract timing features from a race record."""
        
        def parse_length_diff(value):
            """Parse timing differences with 'L' suffix."""
            if not value:
                return None
            try:
                if isinstance(value, str) and value.endswith('L'):
                    return float(value[:-1])
                return float(value)
            except (ValueError, TypeError):
                return None
        
        def parse_time(value):
            """Parse time values in seconds."""
            if not value:
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        timing_data = {
            'overall': None,
            '800m': None,
            '400m': None,
            'final': None,
            'sect_800m': None,
            'sect_400m': None,
            'sect_final': None,
            'finish_time': None
        }
        
        # Check if race is None
        if race is None:
            return timing_data
        
        # Extract from main race record
        if race.get('standardTimeDifference'):
            timing_data['overall'] = parse_length_diff(race.get('standardTimeDifference'))
        
        # Extract from timing object
        timing_obj = race.get('timing', {})
        if timing_obj:
            timing_data['overall'] = timing_data['overall'] or parse_length_diff(timing_obj.get('standardTimeDifference'))
            timing_data['800m'] = parse_length_diff(timing_obj.get('standardTimeTo800Difference'))
            timing_data['400m'] = parse_length_diff(timing_obj.get('standardTime800To400Difference'))
            timing_data['final'] = parse_length_diff(timing_obj.get('standardTime400ToFinishDifference'))
            timing_data['sect_800m'] = parse_time(timing_obj.get('toEightHundredMetresSeconds'))
            timing_data['sect_400m'] = parse_time(timing_obj.get('eightHundredToFourHundredMetresSeconds'))
            timing_data['sect_final'] = parse_time(timing_obj.get('fourHundredToFinishMetresSeconds'))
            timing_data['finish_time'] = parse_time(timing_obj.get('finishTimeSeconds'))
        
        return timing_data
    
    def _calculate_career_context_at_race(self, sorted_races: List[Dict], current_race_idx: int, 
                                        current_race_date: datetime) -> Dict:
        """Calculate career statistics at the time of this historical race."""
        
        prior_races = []
        for i in range(current_race_idx + 1, len(sorted_races)):
            try:
                prior_race = sorted_races[i]
                if prior_race is None:
                    continue
                prior_race_info = prior_race.get('race', {})
                if prior_race_info is None:
                    continue
                prior_race_date_str = prior_race_info.get('date')
                if prior_race_date_str:
                    prior_race_date = self._safe_date_parse(prior_race_date_str)
                    if prior_race_date and current_race_date and prior_race_date < current_race_date:
                        prior_races.append(prior_race)
            except Exception as e:
                logger.warning(f"Error processing prior race {i}: {e}")
                continue
        
        starts = len(prior_races)
        wins = 0
        recent_positions = []
        
        for i, race in enumerate(prior_races):
            try:
                if race is None:
                    continue
                finish_pos = self._parse_finish_position(race.get('finish'))
                if finish_pos:
                    if finish_pos == 1:
                        wins += 1
                    if i < 3:
                        recent_positions.append(finish_pos)
            except Exception as e:
                logger.warning(f"Error processing finish position: {e}")
                continue
        
        win_rate = wins / starts if starts > 0 else 0.0
        recent_form = np.mean(recent_positions) if recent_positions else None
        
        return {
            'starts': starts,
            'wins': wins,
            'win_rate': win_rate,
            'recent_form': recent_form
        }
    
    def _calculate_performance_context_at_race(self, sorted_races: List[Dict], current_race_idx: int,
                                             current_race_date: datetime, target_venue: str, 
                                             target_distance: int, target_condition: str) -> Dict:
        """Calculate venue/distance/condition performance at the time of this historical race."""
        
        # Get prior races (races that happened before this historical race)
        prior_races = []
        for i in range(current_race_idx + 1, len(sorted_races)):
            try:
                prior_race = sorted_races[i]
                if prior_race is None:
                    continue
                prior_race_info = prior_race.get('race', {})
                if prior_race_info is None:
                    continue
                prior_race_date_str = prior_race_info.get('date')
                if prior_race_date_str:
                    prior_race_date = self._safe_date_parse(prior_race_date_str)
                    if prior_race_date and current_race_date and prior_race_date < current_race_date:
                        prior_races.append(prior_race)
            except Exception as e:
                logger.warning(f"Error processing prior race for performance context: {e}")
                continue
        
        # Calculate venue performance from prior races
        venue_races = []
        for race in prior_races:
            try:
                race_info = race.get('race', {})
                if race_info:
                    venue_info = race_info.get('venue', {})
                    venue = venue_info.get('venueName', 'Unknown') if isinstance(venue_info, dict) else str(venue_info)
                    if venue and target_venue and target_venue.lower() in venue.lower():
                        venue_races.append(race)
            except:
                continue
        
        venue_starts = len(venue_races)
        venue_wins = sum(1 for race in venue_races if self._parse_finish_position(race.get('finish')) == 1)
        venue_places = sum(1 for race in venue_races if self._parse_finish_position(race.get('finish')) and self._parse_finish_position(race.get('finish')) <= 3)
        
        # Calculate distance performance from prior races  
        distance_races = []
        for race in prior_races:
            try:
                race_info = race.get('race', {})
                if race_info:
                    distance = self._safe_int(race_info.get('distance'), 1200)
                    if distance == target_distance:
                        distance_races.append(race)
            except:
                continue
        
        distance_starts = len(distance_races)
        distance_wins = sum(1 for race in distance_races if self._parse_finish_position(race.get('finish')) == 1)
        distance_places = sum(1 for race in distance_races if self._parse_finish_position(race.get('finish')) and self._parse_finish_position(race.get('finish')) <= 3)
        
        # Calculate condition performance from prior races
        condition_races = []
        for race in prior_races:
            try:
                race_info = race.get('race', {})
                if race_info:
                    condition = race_info.get('trackCondition', 'Good')
                    if condition == target_condition:
                        condition_races.append(race)
            except:
                continue
        
        condition_starts = len(condition_races)
        condition_wins = sum(1 for race in condition_races if self._parse_finish_position(race.get('finish')) == 1)
        condition_places = sum(1 for race in condition_races if self._parse_finish_position(race.get('finish')) and self._parse_finish_position(race.get('finish')) <= 3)
        
        # Calculate average finishes
        venue_finishes = [self._parse_finish_position(race.get('finish')) for race in venue_races]
        venue_finishes = [f for f in venue_finishes if f is not None]
        
        distance_finishes = [self._parse_finish_position(race.get('finish')) for race in distance_races]
        distance_finishes = [f for f in distance_finishes if f is not None]
        
        condition_finishes = [self._parse_finish_position(race.get('finish')) for race in condition_races]
        condition_finishes = [f for f in condition_finishes if f is not None]
        
        return {
            # Venue performance
            'venue_win_rate': venue_wins / venue_starts if venue_starts > 0 else 0.0,
            'venue_place_rate': venue_places / venue_starts if venue_starts > 0 else 0.0,
            'venue_avg_finish': np.mean(venue_finishes) if venue_finishes else 5.0,
            'venue_experience': np.log1p(venue_starts),
            'venue_recent_form': np.mean(venue_finishes[-3:]) if len(venue_finishes) >= 1 else 5.0,
            
            # Distance performance
            'distance_win_rate': distance_wins / distance_starts if distance_starts > 0 else 0.0,
            'distance_place_rate': distance_places / distance_starts if distance_starts > 0 else 0.0,
            'distance_avg_finish': np.mean(distance_finishes) if distance_finishes else 5.0,
            'distance_experience': np.log1p(distance_starts),
            'distance_recent_form': np.mean(distance_finishes[-3:]) if len(distance_finishes) >= 1 else 5.0,
            
            # Condition performance
            'condition_win_rate': condition_wins / condition_starts if condition_starts > 0 else 0.0,
            'condition_place_rate': condition_places / condition_starts if condition_starts > 0 else 0.0,
            'condition_avg_finish': np.mean(condition_finishes) if condition_finishes else 5.0,
            'condition_experience': np.log1p(condition_starts),
            'condition_recent_form': np.mean(condition_finishes[-3:]) if len(condition_finishes) >= 1 else 5.0
        }
    
    def records_to_dataframe(self, records: List[HistoricalRaceRecord]) -> pd.DataFrame:
        """Convert training records to pandas DataFrame."""
        try:
            data = []
            for record in records:
                record_dict = {
                    'target_race_date': record.target_race_date,
                    'target_venue': record.target_venue,
                    'target_race_number': record.target_race_number,
                    'target_distance': record.target_distance,
                    'target_track_condition': record.target_track_condition,
                    'target_class': record.target_class,
                    
                    'horse_name': record.horse_name,
                    'horse_code': record.horse_code,
                    
                    'race_date': record.race_date,
                    'venue': record.venue,
                    'distance': record.distance,
                    'track_condition': record.track_condition,
                    'track_rating': record.track_rating,
                    'race_class': record.race_class,
                    'field_size': record.field_size,
                    
                    'finish_position': record.finish_position,
                    'won': record.won,
                    'placed': record.placed,
                    
                    'barrier': record.barrier,
                    'weight': record.weight,
                    'jockey': record.jockey,
                    'trainer': record.trainer,
                    'handicap_rating': record.handicap_rating,
                    
                    'standard_time_diff_overall': record.standard_time_diff_overall,
                    'standard_time_diff_800m': record.standard_time_diff_800m,
                    'standard_time_diff_400m': record.standard_time_diff_400m,
                    'standard_time_diff_final': record.standard_time_diff_final,
                    'sectional_800m': record.sectional_800m,
                    'sectional_400m': record.sectional_400m,
                    'final_sectional': record.final_sectional,
                    'finish_time': record.finish_time,
                    
                    'days_to_target_race': record.days_to_target_race,
                    'distance_difference': record.distance_difference,
                    'same_venue': record.same_venue,
                    'same_track_condition': record.same_track_condition,
                    'same_distance': record.same_distance,
                    'same_class': record.same_class,
                    
                    'career_starts_to_date': record.career_starts_to_date,
                    'career_wins_to_date': record.career_wins_to_date,
                    'career_win_rate_to_date': record.career_win_rate_to_date,
                    'recent_form_3_races': record.recent_form_3_races,
                    
                    # Performance context features
                    'venue_win_rate': record.venue_win_rate,
                    'venue_place_rate': record.venue_place_rate,
                    'venue_avg_finish': record.venue_avg_finish,
                    'venue_experience': record.venue_experience,
                    'venue_recent_form': record.venue_recent_form,
                    'distance_win_rate': record.distance_win_rate,
                    'distance_place_rate': record.distance_place_rate,
                    'distance_avg_finish': record.distance_avg_finish,
                    'distance_experience': record.distance_experience,
                    'distance_recent_form': record.distance_recent_form,
                    'condition_win_rate': record.condition_win_rate,
                    'condition_place_rate': record.condition_place_rate,
                    'condition_avg_finish': record.condition_avg_finish,
                    'condition_experience': record.condition_experience,
                    'condition_recent_form': record.condition_recent_form
                }
                data.append(record_dict)
            
            df = pd.DataFrame(data)
            
            # Add derived columns
            if len(df) > 0:
                df['win_rate_at_distance'] = df.groupby(['horse_name', 'distance'])['won'].transform('mean')
                df['win_rate_at_venue'] = df.groupby(['horse_name', 'venue'])['won'].transform('mean')
                df['avg_finish_at_distance'] = df.groupby(['horse_name', 'distance'])['finish_position'].transform('mean')
            
            logger.info(f"Created training DataFrame with {len(df)} records and {len(df.columns)} features")
            if len(df) > 0:
                logger.info(f"Horses: {df['horse_name'].nunique()}, Win rate: {df['won'].mean():.1%}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting records to DataFrame: {e}")
            raise