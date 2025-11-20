# Monte Carlo Soccer/Hockey Simulator
## Simulate a game of Soccer/Hockey using MC Methods and Sampling Techniques
Same simulation assuming 2D straight-line motion. 

## Requirements
  *  numpy==1.24.4
      *  A depreciation warning will show up in sampling the gaussian for numpy 1.25+
  *  scipy==1.11.4

## Usage
*__init__.py*
    * Imports
*params.py*
    * User defined parameters
        * Field Dimensions
        * Verbosity
*Hockey.py*
  *  User Controlled Weighting parameters
  * Random Seed
  * Driving Class
  * Sets Geometry and Starting Positions of the Players
  * Actions Include: shoot, pass, strip, and carry
  * Sampling Methods
      * Exponential (time) - Timesteps
      * Exponential (Space) - Shooting & Stripping
      * Planckian (Space) - Passing
      * Gaussian (Speed) - Calcualte distance from speed and time
  * Save Frames and Animate
  * Determine Player Motions
      * With the ball - Move towards opponents goal, shoot, or pass
      * On Offense - Move towards opponents goal
      * On Defense - Move towards ball, move towards their own goal, or strip

## Future Work
  * Parametric Study?
  * Variance?
  * Expand from 8 players xto n players
