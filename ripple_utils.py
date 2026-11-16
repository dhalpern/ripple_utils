# Code developed by David Halpern
# Adapted from John Sakon's SWRmodule https://github.com/jjsakon/johnModules/blob/master/SWRmodule.py, https://github.com/jjsakon/SWR
# Description from Norman et al. (2019)
# Prior to ripple detection, a reference signal from a nearby white-matter contact was subtracted to eliminate common noise. LFPs were then 
# filtered between 70 and 180 Hz (zero-lag linear-phase Hamming windowed FIR filter with a transition bandwidth of 5 Hz) and instantaneous 
# analytic amplitude was computed using a Hilbert transform.
# Following the procedure of (78), extreme values were clipped to 4 SD to minimize ripple rate–induced biasing. The clipped signal was then 
# squared and smoothed (Kaiser-window FIR low-pass filter with 40 Hz cutoff), and the mean and SD were computed across the entire experimental 
# duration to define the threshold for event detection. Events from the original (squared but unclipped) signal that exceeded 4 SD above 
# baseline were selected as candidate SWR events. Event duration was expanded until ripple power fell below 2 SD. Events shorter than 20 ms or 
# longer than 200 ms were excluded. Adjacent events with less than 30 ms separation (peak-to-peak) were merged. Finally, SWR peak was aligned 
# to the trough (of the nonrectified signal) closest to the peak power.

# import copy
# import mne
import scipy as sp
import numpy as np
# from scipy.signal import firwin, filtfilt, kaiserord, convolve2d, hilbert

def getLogicalChunks(array):
    # find start and end indices for chunks of values >0 in an array
    foundstart = False
    foundend = False
    startindex = 0
    endindex = 0
    starts = []
    ends = []    
    for i in range(0, len(array)):
        if array[i] != 0:
            if not foundstart:
                foundstart = True
                startindex = i
        else:
            # EF1208, array[endindex] will equal 0, is that desired?
            if foundstart:
                foundend = True
                endindex = i - 1
        # for last point
        if i==(len(array)-1):
            if foundstart:
                starts.append(startindex)
                # I don't understand this, EF1208
                if array[i] == 1:
                    ends.append(i)
                else:
                    ends.append(i-1)
        if foundend:
            #print(startindex, endindex)
            starts.append(startindex)
            ends.append(endindex)
            foundstart = False
            foundend = False
            startindex = 0
            endindex = 0  
    return starts,ends

def detectRipplesHamming(eeg_rip,trans_width,sr,iedlogic):
    
    '''
    :param array eeg_rip: hilbert transformed ieeg data 
    '''
    # detect ripples similar to with Butterworth, but using Norman et al 2019 algo (based on Stark 2014 algo). Description:
#      Then Hilbert, clip extreme to 4 SD, square this clipped, smooth w/ Kaiser FIR low-pass filter with 40 Hz cutoff,
#      mean and SD computed across entire experimental duration to define the threshold for event detection
#      Events from original (squared but unclipped) signal >4 SD above baseline were selected as candidate SWR events. 
#      Duration expanded until ripple power <2 SD. Events <20 ms or >200 ms excluded. Adjacent events <30 ms separation (peak-to-peak) merged.

    candidate_SD = 3
    artifact_buffer = 100 # ms around IED events to remove SWRs
    sr_factor = 1000 / sr
    ripple_min = 20 / sr_factor # convert each to ms
    ripple_max = 250 / sr_factor #200/sr_factor
    min_separation = 30 / sr_factor # peak to peak
    # orig_eeg_rip = copy.copy(eeg_rip)
    orig_eeg_rip = eeg_rip.copy()
    clip_SD = np.mean(eeg_rip) + candidate_SD * np.std(eeg_rip)
    eeg_rip[eeg_rip>clip_SD] = clip_SD # clip at 3SD since detecting at 3 SD now
    eeg_rip = eeg_rip ** 2 # square
    
    # FIR lowpass 40 hz filter for Norman dtection algo
    nyquist = sr / 2
    ntaps40, beta40 = sp.signal.kaiserord(40, trans_width / nyquist)
    # kaiser_40lp_filter = sp.signal.firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, nyq=nyquist, pass_zero='lowpass')
    kaiser_40lp_filter = sp.signal.firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, fs=sr, pass_zero='lowpass')
    
    eeg_rip = sp.signal.filtfilt(kaiser_40lp_filter,1.,eeg_rip)
    mean_detection_thresh = np.mean(eeg_rip)
    std_detection_thresh = np.std(eeg_rip)
    
    # now, find candidate events (>mean+3SD) 
    orig_eeg_rip = orig_eeg_rip**2
    candidate_thresh = mean_detection_thresh+candidate_SD*std_detection_thresh
    expansion_thresh = mean_detection_thresh+2*std_detection_thresh
    ripplelogic = orig_eeg_rip >= candidate_thresh # EF1208, will evaluate to squared signal is above threshold
    # remove IEDs detected from Norman 25-60 algo...maybe should do this after expansion to 2SD??
    iedlogic = sp.signal.convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- 50 ms from each ied point
    ripplelogic[iedlogic==1] = 0 # EF1208, convert above threshold events to 0 if they are an IED
    
    # expand out to 2SD around surviving events
    num_trials = ripplelogic.shape[0]
    trial_length = ripplelogic.shape[1]
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]
        starts,ends = getLogicalChunks(ripplelogictrial)
        data_trial = orig_eeg_rip[trial]
        for i,start in enumerate(starts):
            current_time = 0
            while data_trial[start+current_time]>=expansion_thresh:
                if (start+current_time)==-1:
                    break
                else:
                    current_time -=1
            starts[i] = start+current_time+1
        for i,end in enumerate(ends):
            current_time = 0
            while data_trial[end+current_time]>=expansion_thresh:
                if (end+current_time)==trial_length-1:
                    break
                else:
                    current_time +=1
            ends[i] = end+current_time
            
        # remove any duplicates from starts and ends
        starts = np.array(starts); ends = np.array(ends)
        _,start_idxs = np.unique(starts, return_index=True)
        _,end_idxs = np.unique(ends, return_index=True)
        starts = starts[start_idxs & end_idxs]
        ends = ends[start_idxs & end_idxs]

        # remove ripples <min or >max length
        lengths = ends-starts
        ripple_keep = (lengths > ripple_min) & (lengths < ripple_max)
        starts = starts[ripple_keep]; ends = ends[ripple_keep]

        # get peak times of each ripple to combine those < 30 ms separated peak-to-peak
        if len(starts)>1:
            max_idxs = np.zeros(len(starts))
            for ripple in range(len(starts)):
                max_idxs[ripple] = starts[ripple] + np.argmax(data_trial[starts[ripple]:ends[ripple]])                    
            overlappers = np.where(np.diff(max_idxs)<min_separation)

            if len(overlappers[0])>0:
                ct = 0
                for overlap in overlappers:
                    ends = np.delete(ends,overlap-ct)
                    starts = np.delete(starts,overlap+1-ct)
                    ct+=1 # so each time one is removed can still remove the next overlap
                
        # turn starts/ends into a logical array and replace the trial in ripplelogic
        temp_trial = np.zeros(trial_length)
        for i in range(len(starts)):
            temp_trial[starts[i]:ends[i]]=1
        ripplelogic[trial] = temp_trial # place it back in
    return ripplelogic

def getStartEndArrays(ripple_array, start_only=False):
    '''
    Get separate arrays of SWR starts and SWR ends from the full binarized array
    '''
    
    # Shift the ripple array to the right and left by one position
    shifted_right = np.roll(ripple_array, shift=1, axis=1)
    shifted_left = np.roll(ripple_array, shift=-1, axis=1)
    
    # Find the start by looking for a transition from 0 to 1
    start_array = (ripple_array == 1) & (shifted_right == 0)
    start_array[:, 0] = ripple_array[:, 0]  # Handle the edge case for the first column
    
    # Find the end by looking for a transition from 1 to 0
    end_array = (ripple_array == 1) & (shifted_left == 0)
    end_array[:, -1] = ripple_array[:, -1]  # Handle the edge case for the last column
    
    if start_only:
        return start_array.astype('uint8')
    return start_array.astype('uint8'), end_array.astype('uint8')

#https://github.com/mne-tools/mne-python/blob/d74cc20c2bd958ee05e0b1158fbfe21a2754b406/mne/filter.py#L43
def next_fast_len(target):
    """Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    Copied from SciPy with minor modifications.
    """
    from bisect import bisect_left

    hams = (
        8,
        9,
        10,
        12,
        15,
        16,
        18,
        20,
        24,
        25,
        27,
        30,
        32,
        36,
        40,
        45,
        48,
        50,
        54,
        60,
        64,
        72,
        75,
        80,
        81,
        90,
        96,
        100,
        108,
        120,
        125,
        128,
        135,
        144,
        150,
        160,
        162,
        180,
        192,
        200,
        216,
        225,
        240,
        243,
        250,
        256,
        270,
        288,
        300,
        320,
        324,
        360,
        375,
        384,
        400,
        405,
        432,
        450,
        480,
        486,
        500,
        512,
        540,
        576,
        600,
        625,
        640,
        648,
        675,
        720,
        729,
        750,
        768,
        800,
        810,
        864,
        900,
        960,
        972,
        1000,
        1024,
        1080,
        1125,
        1152,
        1200,
        1215,
        1250,
        1280,
        1296,
        1350,
        1440,
        1458,
        1500,
        1536,
        1600,
        1620,
        1728,
        1800,
        1875,
        1920,
        1944,
        2000,
        2025,
        2048,
        2160,
        2187,
        2250,
        2304,
        2400,
        2430,
        2500,
        2560,
        2592,
        2700,
        2880,
        2916,
        3000,
        3072,
        3125,
        3200,
        3240,
        3375,
        3456,
        3600,
        3645,
        3750,
        3840,
        3888,
        4000,
        4050,
        4096,
        4320,
        4374,
        4500,
        4608,
        4800,
        4860,
        5000,
        5120,
        5184,
        5400,
        5625,
        5760,
        5832,
        6000,
        6075,
        6144,
        6250,
        6400,
        6480,
        6561,
        6750,
        6912,
        7200,
        7290,
        7500,
        7680,
        7776,
        8000,
        8100,
        8192,
        8640,
        8748,
        9000,
        9216,
        9375,
        9600,
        9720,
        10000,
    )

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            p2 = 2 ** int(quotient - 1).bit_length()

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

#https://github.com/mne-tools/mne-python/blob/d74cc20c2bd958ee05e0b1158fbfe21a2754b406/mne/filter.py#L2796C1-L2819C15
def _my_hilbert(x, n_fft=None, envelope=False):
    """Compute Hilbert transform of signals w/ zero padding.

    Parameters
    ----------
    x : array, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.

    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
    n_x = x.shape[-1]
    out = sp.signal.hilbert(x, N=n_fft, axis=-1)[..., :n_x]
    if envelope:
        out = np.abs(out)
    return out

def get_ripplelogic(eeg, sr=None, times=None):
    ## FILTERS ##
    trans_width = 5.
    ntaps = (2 / 3) * np.log10(1 / (10 * (1e-3 * 1e-4))) * (sr / trans_width) # gives 400 with sr=500, trans=5
    if sr == 512 or sr == 1024 or sr == 1023.999: # last one fixes R1221P
        ntaps = np.ceil(ntaps)
    FIR_bandstop = sp.signal.firwin(int(ntaps+1), 
                                    [70.,178.], 
                                    fs=sr, 
                                    window='hamming',
                                    pass_zero='bandstop')
    bandstop_25_60 = sp.signal.firwin(int(ntaps+1), 
                                      [20.,58.], 
                                      fs=sr, 
                                      window='hamming',
                                      pass_zero='bandstop') # Norman 2019 IED
    nyquist = sr / 2
    ntaps40, beta40 = sp.signal.kaiserord(40, trans_width / nyquist)
    kaiser_40lp_filter = sp.signal.firwin(ntaps40, 
                                          cutoff=40, 
                                          window=('kaiser', beta40), 
                                          scale=False, 
                                          # nyq=nyquist, #deprecated
                                          fs=sr,
                                          pass_zero='lowpass')
    
    # https://github.com/mne-tools/mne-python/blob/maint/1.8/mne/filter.py#L2671-L2782
    n_fft = next_fast_len(len(times))
    
    eeg_rip = eeg.copy()
    eeg_ied = eeg.copy()
    eeg_rip = eeg_rip-sp.signal.filtfilt(FIR_bandstop,
                                                   1.,
                                                   eeg_rip)
    eeg_ied = eeg_ied-sp.signal.filtfilt(bandstop_25_60,
                                                   1.,
                                                   eeg_ied)
    eeg_rip = _my_hilbert(eeg_rip, envelope=True, n_fft=n_fft)
    eeg_ied = _my_hilbert(eeg_ied, envelope=True, n_fft=n_fft)
    
    # select detection algorithm (note that iedlogic is same for both so always run that)
    eeg_ied = eeg_ied**2 # already rectified now square
    eeg_ied = sp.signal.filtfilt(kaiser_40lp_filter,1.,eeg_ied) # low pass filter  
    mean1 = np.mean(eeg_ied)
    std1 = np.std(eeg_ied)
    iedlogic = eeg_ied>=mean1+4*std1 # Norman et al 2018     

    # detect ripples
    ripplelogic = detectRipplesHamming(eeg_rip,trans_width,sr,iedlogic)   

    return ripplelogic