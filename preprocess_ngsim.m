%% Process NGSIM dataset 
% This is required before training the models in our work: 
% "Maneuver-Aware Pooling for Vehicle Trajectory Prediction", submitted to IROS' 21
% as part of our project on https://github.com/m-hasan-n/pooling
% and inspired by https://github.com/nachiket92/conv-social-pooling/blob/master/preprocess_data.m
% Mohamed Hasan

function preprocess_ngsim(us_dataset_dir, i80_dataset_dir)

% NGSIM dataset has 6 subsets 
n_subsets = 6;

% Threshold on acceleration used to compute the longitudinal acceleration 
acc_thresh = 0.7;

% Future horizon
T_f = 50;

%% Inputs:
% Locations of raw input files:
us101_1 = fullfile(us_dataset_dir,...
    '0750am-0805am\trajectories-0750am-0805am.txt');
us101_2 = fullfile(us_dataset_dir,...
    '0805am-0820am\trajectories-0805am-0820am.txt');
us101_3 = fullfile(us_dataset_dir,...
    '0820am-0835am\trajectories-0820am-0835am.txt');

i80_1 = fullfile(i80_dataset_dir,...
    '0400pm-0415pm\trajectories-0400-0415.txt');
i80_2 = fullfile(i80_dataset_dir,...
    '0500pm-0515pm\trajectories-0500-0515.txt');
i80_3 = fullfile(i80_dataset_dir,...
    '0515pm-0530pm\trajectories-0515-0530.txt');

%% Dataset Fields: 
%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X (lateral Pos)
5: Local Y (longitude Pos)
6: Lane Id
7: Vehicle Acceleartion
8: Vehicle Velocity
9: Lateral maneuver class
10: Longitudinal maneuver class
11-49: Neighbor Car Ids at grid location
%}

ds_ind = 1;
veh_ind = 2;
frame_ind = 3;
lane_ind = 6;
acc_ind = 7;
vel_ind = 8;
lat_class_ind = 9;
lon_class_ind = 10;
nbr_base_ind = 11;

%% Load data and add dataset ids
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

%
vehTrajs = cell(n_subsets, 1);
vehTimes = cell(n_subsets, 1);
for k = 1:n_subsets

    traj{k} = traj{k}(:,[1,2,3,6,7,15, 14 13]);
    
    %correct lane id for US101
    if k <=3
        traj{k}(traj{k}(:,6)>=6,6) = 6;
    end
    
    %Initialization
    vehTrajs{k} = containers.Map;
    vehTimes{k} = containers.Map;
end

%% Parse fields:

for ii = 1 : n_subsets
    
    vehIds = unique(traj{ii}(:,veh_ind));
    for v = 1:length(vehIds)
        vehicle_data = traj{ii}(traj{ii}(:,veh_ind) == vehIds(v),:);
        vehTrajs{ii}(int2str(vehIds(v))) = vehicle_data;
        
        % Find lateral and longitudinal intentions 
        [vehicle_lat_int, vehicle_lon_int]= find_intention(vehicle_data, T_f, acc_thresh, lane_ind, acc_ind);        
        
        traj{ii}(traj{ii}(:,veh_ind) == vehIds(v), lat_class_ind) = vehicle_lat_int;
        traj{ii}(traj{ii}(:,veh_ind) == vehIds(v), lon_class_ind) = vehicle_lon_int;
            
    end
    
    timeFrames = unique(traj{ii}(:,frame_ind));
    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,frame_ind) == timeFrames(v),:);
    end
    
    for k = 1:length(traj{ii}(:,1))        
        
        % Get grid locations of the neighbor vehicles
        % 5: Local Y (longitude Pos)
        time = traj{ii}(k,frame_ind);
        lane = traj{ii}(k,lane_ind);
        t = vehTimes{ii}(int2str(time));
        
        frameEgo = t(t(:,lane_ind) == lane,:);
        frameL = t(t(:,lane_ind) == lane-1,:);
        frameR = t(t(:,lane_ind) == lane+1,:);
        
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 1+round((y+90)/15);
                    traj{ii}(k,nbr_base_ind-1+gridInd) = frameL(l,2);
                end
            end
        end
        
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,nbr_base_ind-1+gridInd) = frameEgo(l,2);
            end
        end
        
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,nbr_base_ind-1+gridInd) = frameR(l,2);
                end
            end
        end
        
    end
end


%% Split train, validation, test

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6}];
clear traj;

trajTr = [];
trajVal = [];
trajTs = [];

for k = 1:6
    ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,2)));
    ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,2)));
    
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
    
end

 tracksTr = {};
for k = 1:6
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l), [3:5 vel_ind])';
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:6
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l), [3:5 vel_ind])';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:6
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l), [3:5 vel_ind])';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end


%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, 
% the initial 3 seconds of each trajectory is not used for training/testing

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);

indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracksVal{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>t+1
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);

indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracksTs{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>t+1
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);

%% Save mat files:

traj = trajTr;
tracks = tracksTr;
save('data/TrainSet','traj','tracks');

traj = trajVal;
tracks = tracksVal;
save('data/ValSet','traj','tracks');

traj = trajTs;
tracks = tracksTs;
save('data/TestSet','traj','tracks');

%% Save subsets of the test data to perform the manuever-based evaluation
mnvr_eval_subsets(trajTs, tracksTs, ds_ind, lane_ind, lat_class_ind)

end

%% Helper Function

% Lateral Intention
%1 (Keep), 2(Left), 3(Right)
function [vehicle_lat_int, vehicle_lon_int ]= find_intention(vehicle_data, T_f, acc_thresh, lane_ind, acc_ind)

N_frames = size(vehicle_data,1);
vehicle_lat_int = zeros(N_frames,1);
vehicle_lon_int = zeros(N_frames,1);

for jj = 1 : N_frames
    
    %indices of history and future trajectories
    fut_ind = jj+2 : min(N_frames, jj+T_f+1);
    
    %vehicle mean future acceleration 
    ego_fut_acc = vehicle_data(fut_ind, acc_ind);
    aFut = mean(ego_fut_acc);
    
    %Current lane
    curr_lane = vehicle_data(jj,lane_ind);
    
    %find intentions
    if length(fut_ind) < 2
        vehicle_lat_int(jj) = 1;
        vehicle_lon_int(jj) = 1;
    else
        fut_lane = vehicle_data(fut_ind(end),lane_ind);
        vehicle_lat_int(jj) = set_lat_manuver(curr_lane, fut_lane);
         vehicle_lon_int(jj) = set_lon_intention(aFut, acc_thresh); 
    end
    
end
end

%lateral intention: 1 (Keep), 2(Left), 3(Right)
function lat_intention = set_lat_manuver(curr_lane, fut_lane)

if fut_lane > curr_lane
    lat_intention = 3;
elseif fut_lane < curr_lane
    lat_intention = 2;
else
    lat_intention = 1;
end
      
end

% Longitudinal Intention
% 1 (normal speed), 2 (deceleration) 3(acceleration)
function lon_int = set_lon_intention(aFut, acc_thresh)

if aFut < -acc_thresh
    lon_int = 2;
elseif aFut> acc_thresh
    lon_int = 3;
else
    lon_int = 1;
end
end

%Select subsets of the test data perform the manuever-based evaluation
%1. Arbitrary Left Lane Change 
%2. Compulsory Left Lane Change 'Merging' 
%3. Right Lane Change 
%4. keep lane 
function mnvr_eval_subsets(traj, tracks, ds_ind, lane_ind, lat_class_ind)

left_traj = [];
right_traj = [];
merge_traj = [];
keep_traj=[];

%Iterate on the two subsets of NGSIM (I80 and US101)
for ii = 1 : 2
      
    %dataset ID
    if ii==1
        % I80 subset
        ds_ids = traj(:,ds_ind)==4 | traj(:,ds_ind)==5 | traj(:,ds_ind)==6;
        merging_ids = traj(:,lane_ind) ==7;
    else
        % US101 subset
        ds_ids = traj(:,ds_ind)== 1 | traj(:,ds_ind)==2 | traj(:,ds_ind)==3;
        merging_ids = traj(:,lane_ind) ==6;
    end
    
    %IDs of lateral maneuvers
    lat_intention = traj(:, lat_class_ind);
    keep_lane = lat_intention== 1;
    lc_left = lat_intention== 2;
    lc_right = lat_intention == 3;
    
    %1. Arbitrary Left LC
    arb_left_ids = ds_ids&lc_left&~merging_ids;
    arb_left_traj = traj(arb_left_ids,:);
    
    %2. Compulsory Left LC 'Merging'
    merge_ids = ds_ids&lc_left&merging_ids;
    merging_traj = traj(merge_ids,:);
    
    %3. Right LC
    lc_right_ids =  ds_ids&lc_right;
    lc_right_traj = traj(lc_right_ids,:);
    
    %4. Keep Lane
    keeping_ids = ds_ids&keep_lane;
    keeping_traj = traj(keeping_ids ,:);
    
    
    assert( (size(arb_left_traj,1) +  size(merging_traj,1) + size(lc_right_traj,1) +...
        size(keeping_traj,1) ) == sum(ds_ids) )
    
    left_traj = [left_traj;arb_left_traj];
    right_traj = [right_traj;lc_right_traj];
    merge_traj = [merge_traj;merging_traj];
    keep_traj=[keep_traj;keeping_traj];

end

%Saving
fname = 'data/TestSet_left.mat'; 
traj = left_traj;
save(fname,'traj','tracks')

fname = 'data/TestSet_merge.mat';
traj = merge_traj;
save(fname,'traj','tracks')

fname = 'data/TestSet_right.mat';
traj = right_traj;
save(fname,'traj','tracks')

fname = 'data/TestSet_keep.mat';
traj = keep_traj;
save(fname,'traj','tracks')

end