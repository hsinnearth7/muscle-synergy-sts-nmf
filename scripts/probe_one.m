% Probe one subject's RawData.mat / ProcessedData.mat with Octave,
% print the actual shapes of the EMG tables.
pkg load io;  % for loading MAT tables if needed

disp('--- Loading RawData.mat for S001 ---');
raw = load('Gait120_001_to_010/S001/EMG/RawData.mat');
flds = fieldnames(raw);
for i = 1:numel(flds)
  printf('  %s\n', flds{i});
end
printf('\nEMGs_info.fs = %d\n', raw.EMGs_info.fs);

sts = raw.SitToStand;
printf('SitToStand.nTrials = %d\n', sts.nTrials);
printf('SitToStand.AvailableTrialIdx = ');
disp(sts.AvailableTrialIdx);

printf('\n--- Trial01 contents ---\n');
t1 = sts.Trial01;
t1_flds = fieldnames(t1);
for i = 1:numel(t1_flds)
  printf('  %s\n', t1_flds{i});
end

printf('\n--- Trial01.EMGs_raw ---\n');
em = t1.EMGs_raw;
printf('class: %s\n', class(em));
try
  sz = size(em);
  printf('size: %s\n', mat2str(sz));
catch
end

% If it is a table, list variable names
if strcmp(class(em), 'table')
  printf('columns: %s\n', strjoin(em.Properties.VariableNames, ', '));
  % Convert to cell array and then to matrix
  M = table2array(em);
  printf('table2array size: %s  class: %s\n', mat2str(size(M)), class(M));
end

printf('\nTotalFrame: %s\n', mat2str(t1.TotalFrame));
printf('Step01.TargetFrame: %s\n', mat2str(t1.Step01.TargetFrame));

% MVC info
printf('\n--- Trial01 (ProcessedData) MVCs ---\n');
try
  proc = load('Gait120_001_to_010/S001/EMG/ProcessedData.mat');
  mvcs = proc.SitToStand.Trial01.MVCs;
  printf('MVCs: %s\n', mat2str(mvcs));
catch err
  disp(err.message);
end
