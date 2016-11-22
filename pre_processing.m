


%% Input

filename_wind = 'data/test/wind/wind1.wav'
filename_speech = 'data/test/male1.wav'

SNRin = -0;

%% output

filename_mixture = 'mixture_male_0db.wav';
filename_speech_out = 'male_clean.wav';

%% load files
[wn, fs]=audioread(filename_wind);
[x, fs]=audioread(filename_speech);



L1 = length(wn);
L2 = length(x);

wn = wn(1:L2); % crop signal

% normalize:
% x = x/max(abs(x));
wn = wn * sqrt(sum(x.^2)/(sum(wn.^2)*10^(SNRin/10)));
 
y = x + wn; % Noisy signal

y = y/max(abs(y));

% SNRout
SNRout = 10* log10(sum(x.^2)/sum(wn.^2))


%% write 

audiowrite(filename_mixture, y, fs);
audiowrite(filename_speech_out, x, fs);

sound(y,fs)
