/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(const double x, const double y, const double theta, const double std[]) {
    particles.resize(num_particles);
    weights.resize(num_particles);

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (auto &p : particles)
    {
        p.x = dist_x(engine);
        p.y = dist_y(engine);
        p.theta = dist_theta(engine);
        p.weight = 1;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate) {
    // random gaussian noise around 0 (width given by sensor)
    std::normal_distribution<double> N_x(0, std_pos[0]);
    std::normal_distribution<double> N_y(0, std_pos[1]);
    std::normal_distribution<double> N_theta(0, std_pos[2]);

    // update all particles
    for (auto &p : particles)
    {
        // 1. step - add meassurement
        if (fabs(yaw_rate) < 0.0001)
        {
            // do not consider yaw -> constant velocity, no update to theta
            p.x += delta_t * velocity * cos(p.theta);
            p.y += delta_t * velocity * sin(p.theta);
        }
        else
        {
            // full update of x, y, theta
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        // 2. step - add noise to all values
        p.x += N_x(engine);
        p.y += N_y(engine);
        p.theta += N_theta(engine);
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for (auto &observation : observations)
    {
        double minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < predicted.size(); ++i)
        {
            const double distance = dist(observation.x, observation.y, predicted[i].x, predicted[i].y);
            if (distance < minDistance)
            {
                minDistance = distance;
                observation.id = i;
            }
        }
    }
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    for (size_t i = 0; i < num_particles; ++i)
    {
        auto &p = particles[i];

        // collect landmarks in sensor range
        std::vector<LandmarkObs> predictedLandmarks;
        for (auto &l : map_landmarks.landmark_list)
            if (dist(p.x, p.y, l.x_f, l.y_f) < sensor_range)
                predictedLandmarks.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});

        // convert coordinates of observations to map coordinates
        vector<LandmarkObs> observationsInMapCoordinates;
        for(auto &o : observations)
        {
            LandmarkObs oMap;
            oMap.x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
            oMap.y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
            observationsInMapCoordinates.push_back(oMap);
        }

        // find matching landmarks
        dataAssociation(predictedLandmarks, observationsInMapCoordinates);

        // compute particles weight
        p.weight = 1.0;
        for(auto &oMap: observationsInMapCoordinates)
        {
            const LandmarkObs &landmark = predictedLandmarks[oMap.id];
            const double x_term = pow(oMap.x - landmark.x, 2) / (2 * pow(std_landmark[0], 2));
            const double y_term = pow(oMap.y - landmark.y, 2) / (2 * pow(std_landmark[1], 2));
            p.weight *= exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        }

        // update weight in list
        weights[i] = p.weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
