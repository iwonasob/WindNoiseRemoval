


%% Input
% 
% filename_wind = '../test_old/wind/wind1.wav'
% filename_speech = '../test_old/male1.wav'

filename_wind = 'data/test_raw/wind/wind1.wav'
filename_speech = 'data/test_raw/female1.wav'

SNRin = -5;

%% output

filename_mixture = 'mixture_male_0db.wav';
filename_speech_out = 'male_clean.wav';

%% load files
[wn, fs]=audioread(filename_wind);
[x, fs]=audioread(filename_speech);

x = x/max(abs(x));

L1 = length(wn);
L2 = length(x);

wn = wn(1:L2); % crop signal

sqrt(sum(x.^2)/(sum(wn.^2)*10^(SNRin/10)))
sum(x.^2)
sum(wn.^2)

% normalize:
% x = x/max(abs(x));
wn = wn * sqrt(sum(x.^2)/(sum(wn.^2)*10^(SNRin/10)));
 
y = x + wn; % Noisy signal

% y = y/max(abs(y));

% SNRout
SNRout = 10* log10(sum(x.^2)/sum(wn.^2))


%% write 
% 
% audiowrite(filename_mixture, y, fs);
% audiowrite(filename_speech_out, x, fs);

sound(y,fs)
