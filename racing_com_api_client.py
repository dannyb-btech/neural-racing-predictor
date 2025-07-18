#!/usr/bin/env python3
"""
Racing.com GraphQL API Client

This module provides a Python client for racing.com's GraphQL API,
matching the exact endpoints and queries from your JavaScript implementation.
"""

import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RacingComAPIClient:
    """
    Python client for racing.com's GraphQL API.
    
    This client matches your JavaScript implementation and supports:
    - Getting meetings by month
    - Getting races for a meeting  
    - Getting race entries/field
    - Getting horse form history
    - Getting speed maps
    """
    
    # GraphQL API URLs
    BASE_URL = 'https://graphql.rmdprod.racing.com/'
    SPEED_MAP_BASE_URL = 'https://fais-api.racing.com/api/tippinghub/graphql/'
    
    # GraphQL Queries (copied from your JavaScript)
    QUERIES = {
        'GET_MEETINGS_BY_MONTH': """
        query GetMeetingByMonth_CD($year: Int!, $month: Int!) {
            GetMeetingByMonth(year: $year, month: $month) {
                id
                venue
                venueAbbr
                venueCode
                date
                state
                meetCode: id
                meetUrl
                status
                isJumpOut
                isTrial
                races {
                    raceNumber
                    name
                    status
                    distance
                    group
                    class
                    nameForm
                }
            }
        }
        """,
        
        'GET_RACES_FOR_MEET': """
        query getRaceNumberList_CD($meetCode: ID!) {
            getRacesForMeet(meetCode: $meetCode) {
                id
                meet {
                    venue
                    meetUrl
                    meetUrlSegment
                }
                raceNumber
                raceStatus
                distance
                time
                name
                nameForm
                trackCondition
                isTrial
                isJumpOut
                trackRating
                hasSpeedMap
                hasResults
                hasTips
            }
        }
        """,
        
        'GET_RACE_ENTRIES': """
        query getRaceEntriesForField_CD($meetCode: ID!, $raceNumber: Int!) {
            getRaceForm(meetCode: $meetCode, raceNumber: $raceNumber) {
                id
                status
                tempo
                isTrial
                isJumpOut
                location
                venueCode
                venueState
                venue {
                    venueName
                }
                distance
                trackCondition
                trackRating
                class
                rdcClass
                group
                nameForm
                bestBets {
                    overview
                    suggestedBet
                    tipCondition
                    selectionTipper
                }
                raceTips {
                    raceCode
                    condition
                    comment
                    tipType
                    tipster {
                        tipsterId
                        tipsterName
                        isLead
                    }
                    tips {
                        position
                        tipBetType
                        comment
                        raceEntryItem {
                            raceEntryNumber
                            horseName
                            horseCode
                            horseCountry
                            barrierNumber
                            trainerName
                            jockeyName
                            apprenticeCanClaim
                            apprenticeAllowedClaim
                            speedValue
                            faisHighlight {
                                key
                                positive
                            }
                            odds {
                                providerCode
                                oddsPlace
                                oddsWin
                                oddsIsFavouriteWin
                                oddsIsMarketMover
                            }
                        }
                    }
                }
                formRaceEntries {
                    id
                    weight
                    weightPrevious
                    comment
                    horseName
                    horseCode
                    horseCountry
                    jockeyName
                    trainerName
                    trainerCode
                    raceEntryNumber
                    barrierNumber
                    scratched
                    emergency
                    apprenticeCanClaim
                    apprenticeAllowedClaim
                    speedValue
                    faisHighlight {
                        key
                        positive
                    }
                    odds {
                        providerCode
                        oddsPlace
                        oddsWin
                        oddsIsFavouriteWin
                        oddsIsMarketMover
                    }
                    lastGear
                    gearChanges
                    gearHasChanges
                    handicapRating
                    handicapRatingProgression
                    trackDistanceStats
                    trackStats
                    distanceStats
                    jockeyStats
                    atThisClassStats
                    lastRaceDate
                    horse {
                        id
                        age
                        sex
                        colour
                        rating
                        ratingProgression
                        careerWinPercent
                        careerPlacePercent
                        lastFive
                        country
                        owners
                        sireHorseName
                        damHorseName
                    }
                }
            }
        }
        """,
        
        'GET_HORSE_FORM': """
        query getRaceEntryItemByHorsePaged_CD($horseCode: ID!, $lastEvaluatedKey: String) {
            GetRaceEntryItemByHorsePaged(horseCode: $horseCode, limit: 10, lastEvaluatedKey: $lastEvaluatedKey) {
                id
                finish
                finishAbv
                horseName
                horseCountry
                trainerName
                raceEntryNumber
                race {
                    id
                    date
                    isTrial
                    isJumpOut
                    venue {
                        venueName
                    }
                    venueCode
                    distance
                    raceNumber
                    class
                    trackCondition
                    trackRating
                    runnersCount
                    group
                    nameForm
                    totalPrizeMoney
                    raceTime
                    standardTimeDifference
                    toEightHundredMetresSeconds
                    eightHundredToFourHundredMetresSeconds
                    fourHundredToFinishMetresSeconds
                }
                weightCarried
                jockeyName
                barrierNumber
                margin
                startingPrice
                winningTime
                positionAtSettledAbv
                positionAt800Abv
                positionAt400Abv
                standardTimeDifference
                timeSectional
                comment
                commentStewards
                handicapRating
                timing {
                    toEightHundredMetresSeconds
                    standardTimeTo800Difference
                    eightHundredToFourHundredMetresSeconds
                    standardTime800To400Difference
                    fourHundredToFinishMetresSeconds
                    standardTime400ToFinishDifference
                    finishTimeSeconds
                    standardTimeDifference
                }
                LastEvaluatedKey
            }
        }
        """,
        
        'GET_SPEED_MAPS': """
        query GetSpeedMaps($meetCode: ID!, $masterEventId: ID!) {
            speedMaps: earlySpeedInMeet(meetCode: $meetCode, masterEventId: $masterEventId) {
                eventId
                raceNumber
                tempo
                railInformation
                hasEarlySpeedValues
                raceEntries {
                    horseName
                    barrierNumber
                    speedValue
                    isScratched
                    winPrice {
                        price
                    }
                }
            }
        }
        """,
        
        'GET_JOCKEY_PROFILE': """
        query getJockeyProfile($id: ID!) {
            getJockeyProfile(id: $id) {
                id
                age
                ridingWeight
                sitecoreId
                firstName
                lastName
                fullName
                careerWins
                group1Wins
                winPercent
                recentWinPercent
                prizeMoney
                mostRidesForTrainerCode
                firstRideRaceEntryItem {
                    id
                    horseName
                    raceDate
                    __typename
                }
                __typename
            }
        }
        """,
        
        'GET_TRAINER_PROFILE': """
        query getTrainerProfile($id: ID!) {
            getTrainerProfile(id: $id) {
                id
                sitecoreId
                location
                fullName
                careerWins
                group1Wins
                winPercent
                placePercent
                based
                prizeMoney
                recentWinPercent
                firstWinRaceEntryItem {
                    id
                    horseName
                    raceDate
                    __typename
                }
                __typename
            }
        }
        """
    }
    
    def __init__(self, api_key: str, rate_limit_requests: int = 100, rate_limit_window: int = 60):
        """
        Initialize the Racing.com API client.
        
        Args:
            api_key: Racing.com API key (x-api-key header)
            rate_limit_requests: Number of requests allowed per window
            rate_limit_window: Time window for rate limiting (seconds)
        """
        self.api_key = api_key
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        
        # Rate limiting tracking
        self.request_times = []
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Set headers for racing.com API
        self.session.headers.update({
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Bayesian-Racing-Model/1.0'
        })
    
    def _rate_limit_check(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests outside the current window
        self.request_times = [
            req_time for req_time in self.request_times 
            if current_time - req_time < self.rate_limit_window
        ]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.rate_limit_requests:
            sleep_time = self.rate_limit_window - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    def _make_graphql_request(self, base_url: str, query: str, variables: Dict, retries: int = 3) -> Dict:
        """
        Make a GraphQL request to racing.com API.
        
        Args:
            base_url: Base URL (main API or speed map API)
            query: GraphQL query string
            variables: Query variables
            retries: Number of retry attempts
            
        Returns:
            GraphQL response data
        """
        self._rate_limit_check()
        
        payload = {
            'query': query,
            'variables': variables
        }
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Making GraphQL request to {base_url} (attempt {attempt + 1})")
                
                response = self.session.post(base_url, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for GraphQL errors
                if 'errors' in data:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    raise Exception(f"GraphQL errors: {data['errors']}")
                
                return data.get('data', {})
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt == retries:
                    logger.error(f"All retry attempts failed for {base_url}")
                    raise
                
                # Exponential backoff
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        raise Exception("Should not reach here")
    
    def get_meetings_by_month(self, year: int, month: int) -> List[Dict]:
        """
        Get racing meetings for a specific year and month.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            List of meetings (filtered to exclude jumpouts and trials)
        """
        try:
            logger.info(f"Fetching meetings for {year}-{month:02d}")
            
            variables = {'year': year, 'month': month}
            data = self._make_graphql_request(
                self.BASE_URL, 
                self.QUERIES['GET_MEETINGS_BY_MONTH'], 
                variables
            )
            
            meetings = data.get('GetMeetingByMonth', [])
            
            # Filter out jumpouts and trials (matching your JavaScript logic)
            filtered_meetings = [
                meeting for meeting in meetings 
                if not meeting.get('isJumpOut') and not meeting.get('isTrial')
            ]
            
            logger.info(f"Found {len(filtered_meetings)} meetings for {year}-{month:02d}")
            return filtered_meetings
            
        except Exception as e:
            logger.error(f"Error fetching meetings for {year}-{month}: {e}")
            raise
    
    def get_races_for_meet(self, meet_code: str) -> List[Dict]:
        """
        Get races for a specific meeting.
        
        Args:
            meet_code: Meeting code/ID
            
        Returns:
            List of races sorted by race number
        """
        try:
            logger.info(f"Fetching races for meeting {meet_code}")
            
            variables = {'meetCode': meet_code}
            data = self._make_graphql_request(
                self.BASE_URL,
                self.QUERIES['GET_RACES_FOR_MEET'],
                variables
            )
            
            races = data.get('getRacesForMeet', [])
            
            # Sort by race number (matching your JavaScript logic)
            sorted_races = sorted(races, key=lambda x: x.get('raceNumber', 0))
            
            logger.info(f"Found {len(sorted_races)} races for meeting {meet_code}")
            return sorted_races
            
        except Exception as e:
            logger.error(f"Error fetching races for meeting {meet_code}: {e}")
            raise
    
    def get_race_entries(self, meet_code: str, race_number: int) -> Dict:
        """
        Get race entries (field) for a specific race.
        
        Args:
            meet_code: Meeting code/ID
            race_number: Race number
            
        Returns:
            Race form data with entries
        """
        try:
            logger.info(f"Fetching entries for race {race_number} at meeting {meet_code}")
            
            variables = {'meetCode': meet_code, 'raceNumber': race_number}
            data = self._make_graphql_request(
                self.BASE_URL,
                self.QUERIES['GET_RACE_ENTRIES'],
                variables
            )
            
            race_form = data.get('getRaceForm', {})
            
            num_entries = len(race_form.get('formRaceEntries', []))
            logger.info(f"Found {num_entries} entries for race {race_number}")
            
            return race_form
            
        except Exception as e:
            logger.error(f"Error fetching entries for race {race_number} at {meet_code}: {e}")
            raise
    
    def get_horse_form(self, horse_code: str, max_races: int = 100) -> List[Dict]:
        """
        Get complete horse form history by paginating through all available races.
        
        Args:
            horse_code: Horse code/ID
            max_races: Maximum number of races to fetch (safety limit)
            
        Returns:
            Complete list of horse race entries
        """
        try:
            logger.info(f"Fetching complete form history for horse {horse_code}")
            
            all_races = []
            last_evaluated_key = None
            page_count = 0
            max_pages = max_races // 10 + 1  # Each page has limit of 10
            
            while page_count < max_pages:
                variables = {
                    'horseCode': horse_code,
                    'lastEvaluatedKey': last_evaluated_key
                }
                
                data = self._make_graphql_request(
                    self.BASE_URL,
                    self.QUERIES['GET_HORSE_FORM'],
                    variables
                )
                
                page_races = data.get('GetRaceEntryItemByHorsePaged', [])
                
                if not page_races:
                    # No more races available
                    break
                
                # Add races from this page
                all_races.extend(page_races)
                page_count += 1
                
                # Check if there's a next page
                # The last race entry should contain LastEvaluatedKey for pagination
                last_race = page_races[-1] if page_races else None
                if last_race and 'LastEvaluatedKey' in last_race:
                    last_evaluated_key = last_race['LastEvaluatedKey']
                    if not last_evaluated_key:  # Empty key means no more pages
                        break
                    logger.debug(f"Fetching next page for horse {horse_code} (page {page_count + 1})")
                else:
                    # No pagination key means this is the last page
                    break
                
                # Safety check - if we got less than the limit, likely the last page
                if len(page_races) < 10:
                    break
            
            logger.info(f"Found {len(all_races)} total form entries for horse {horse_code} across {page_count} pages")
            return all_races
            
        except Exception as e:
            logger.error(f"Error fetching form for horse {horse_code}: {e}")
            raise
    
    def get_speed_maps(self, meet_code: str, master_event_id: str) -> List[Dict]:
        """
        Get speed maps for a meeting.
        
        Args:
            meet_code: Meeting code/ID  
            master_event_id: Master event ID
            
        Returns:
            Speed map data
        """
        try:
            logger.info(f"Fetching speed maps for meeting {meet_code}, event {master_event_id}")
            
            variables = {'meetCode': meet_code, 'masterEventId': master_event_id}
            data = self._make_graphql_request(
                self.SPEED_MAP_BASE_URL,
                self.QUERIES['GET_SPEED_MAPS'],
                variables
            )
            
            speed_maps = data.get('speedMaps', [])
            
            logger.info(f"Found {len(speed_maps)} speed maps")
            return speed_maps
            
        except Exception as e:
            logger.error(f"Error fetching speed maps for {meet_code}: {e}")
            raise
    
    def get_jockey_profile(self, jockey_id: str) -> Dict:
        """
        Get jockey profile with career statistics.
        
        Args:
            jockey_id: Jockey ID
            
        Returns:
            Jockey profile data with career statistics
        """
        try:
            logger.info(f"Fetching jockey profile for ID {jockey_id}")
            
            variables = {'id': jockey_id}
            data = self._make_graphql_request(
                self.BASE_URL,
                self.QUERIES['GET_JOCKEY_PROFILE'],
                variables
            )
            
            jockey_profile = data.get('getJockeyProfile', {})
            
            logger.info(f"Successfully fetched profile for jockey {jockey_profile.get('fullName', jockey_id)}")
            return jockey_profile
            
        except Exception as e:
            logger.error(f"Error fetching jockey profile for ID {jockey_id}: {e}")
            raise
    
    def get_trainer_profile(self, trainer_id: str) -> Dict:
        """
        Get trainer profile with career statistics.
        
        Args:
            trainer_id: Trainer ID
            
        Returns:
            Trainer profile data with career statistics
        """
        try:
            logger.info(f"Fetching trainer profile for ID {trainer_id}")
            
            variables = {'id': trainer_id}
            data = self._make_graphql_request(
                self.BASE_URL,
                self.QUERIES['GET_TRAINER_PROFILE'],
                variables
            )
            
            trainer_profile = data.get('getTrainerProfile', {})
            
            logger.info(f"Successfully fetched profile for trainer {trainer_profile.get('fullName', trainer_id)}")
            return trainer_profile
            
        except Exception as e:
            logger.error(f"Error fetching trainer profile for ID {trainer_id}: {e}")
            raise
    
    def get_complete_race_data(self, meet_code: str, race_number: int) -> Tuple[Dict, Dict]:
        """
        Get complete race data including entries and all horse form.
        
        Args:
            meet_code: Meeting code/ID
            race_number: Race number
            
        Returns:
            Tuple of (race_entries, horse_forms_dict)
        """
        try:
            logger.info(f"Fetching complete data for race {race_number} at {meet_code}")
            
            # Get race entries first
            race_entries = self.get_race_entries(meet_code, race_number)
            
            # Extract horse codes from entries, excluding scratched horses
            form_race_entries = race_entries.get('formRaceEntries', [])
            active_entries = [entry for entry in form_race_entries if not entry.get('scratched', False)]
            horse_codes = [entry.get('horseCode') for entry in active_entries if entry.get('horseCode')]
            
            if len(form_race_entries) != len(active_entries):
                scratched_count = len(form_race_entries) - len(active_entries)
                logger.info(f"Excluded {scratched_count} scratched horses from data collection")
            
            logger.info(f"Fetching form for {len(horse_codes)} horses")
            
            # Get form for each horse
            horse_forms = {}
            for horse_code in horse_codes:
                try:
                    horse_form = self.get_horse_form(horse_code)
                    if horse_form:
                        # Use horse name as key (from first entry if available)
                        horse_name = horse_form[0].get('horseName', horse_code) if horse_form else horse_code
                        horse_forms[horse_name] = {
                            'horseCode': horse_code,
                            'horseName': horse_name,
                            'data': {
                                'GetRaceEntryItemByHorsePaged': horse_form
                            }
                        }
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch form for horse {horse_code}: {e}")
                    continue
            
            logger.info(f"Successfully fetched complete data for {len(horse_forms)} horses")
            
            return race_entries, horse_forms
            
        except Exception as e:
            logger.error(f"Error fetching complete race data: {e}")
            raise
    
    def get_upcoming_races_today(self) -> List[Dict]:
        """
        Get upcoming races for today.
        
        Returns:
            List of upcoming races
        """
        try:
            today = datetime.now()
            year = today.year
            month = today.month
            
            logger.info(f"Fetching today's races for {year}-{month:02d}")
            
            # Get meetings for this month
            meetings = self.get_meetings_by_month(year, month)
            
            # Filter for today's meetings
            today_str = today.strftime('%Y-%m-%d')
            today_meetings = [
                meeting for meeting in meetings 
                if meeting.get('date', '').startswith(today_str)
            ]
            
            # Get races for today's meetings
            upcoming_races = []
            for meeting in today_meetings:
                meet_code = meeting.get('meetCode') or meeting.get('id')
                if meet_code:
                    try:
                        races = self.get_races_for_meet(meet_code)
                        for race in races:
                            race['meeting'] = meeting  # Add meeting context
                        upcoming_races.extend(races)
                    except Exception as e:
                        logger.warning(f"Failed to fetch races for meeting {meet_code}: {e}")
                        continue
            
            logger.info(f"Found {len(upcoming_races)} upcoming races today")
            return upcoming_races
            
        except Exception as e:
            logger.error(f"Error fetching today's races: {e}")
            raise


def create_racing_com_client(api_key: str) -> RacingComAPIClient:
    """
    Create a racing.com API client.
    
    Args:
        api_key: Your racing.com API key
        
    Returns:
        Configured API client
    """
    return RacingComAPIClient(api_key)


def demo_racing_com_api():
    """Demonstrate racing.com API client usage."""
    print("🏇 Racing.com GraphQL API Client Demo")
    print("=" * 50)
    
    # This would use your actual API key
    api_key = "your-racing-com-api-key-here"
    
    try:
        client = create_racing_com_client(api_key)
        
        print("📅 Available API methods:")
        methods = [
            "get_meetings_by_month(year, month)",
            "get_races_for_meet(meet_code)",
            "get_race_entries(meet_code, race_number)",
            "get_horse_form(horse_code)",
            "get_speed_maps(meet_code, master_event_id)",
            "get_jockey_profile(jockey_id)",
            "get_trainer_profile(trainer_id)",
            "get_complete_race_data(meet_code, race_number)",
            "get_upcoming_races_today()"
        ]
        
        for method in methods:
            print(f"  • {method}")
        
        print("\n📋 Example usage:")
        print("  # Get current month meetings")
        print("  meetings = client.get_meetings_by_month(2024, 7)")
        print("  ")
        print("  # Get races for a meeting")
        print("  races = client.get_races_for_meet('5185463')")
        print("  ")
        print("  # Get complete race data")
        print("  race_data, horse_forms = client.get_complete_race_data('5185463', 7)")
        print("  ")
        print("  # Get jockey/trainer profiles")
        print("  jockey_profile = client.get_jockey_profile('852369')")
        print("  trainer_profile = client.get_trainer_profile('12345')")
        
        print("\n🎯 API Integration Status:")
        print("  ✅ Matches your JavaScript GraphQL implementation")
        print("  ✅ All 7 API endpoints implemented (including jockey/trainer profiles)")
        print("  ✅ Rate limiting and error handling included")
        print("  ✅ Compatible with existing Bayesian model")
        print("  🔧 Ready for production with your API key")
        
    except Exception as e:
        print(f"❌ Demo error (expected without real API key): {e}")


if __name__ == "__main__":
    demo_racing_com_api()