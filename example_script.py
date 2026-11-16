import pandas as pd
import numpy as np
import xarray as xr
import ripple_utils as ru
import h5py
base_dir = '/Users/davidhalpern/Documents/GitHub/intracranial_food_choice/'
bvs_data_dir = base_dir + 'BVS_iEEG/'
bvs_ieeg_data_dir = bvs_data_dir + 'iEEG_DataFiltered/'
data_dir = base_dir + 'data/Choices/'
out_dir = base_dir + 'data/ripples/'

binned = False
stim_locked = True
if not binned:
  stim_locked = True

def read_contact(row, h5_file):
    return h5_file['data']['iEEG'][row['Electrode']][row['Site']]['filtered_data'][:][0]

alliEEGDataInfo = pd.read_csv(bvs_data_dir+'alliEEGDataInfo.csv')
InfoSub = pd.read_csv(bvs_data_dir+'InfoSub.csv') # sample rate
alliEEGDataInfo = alliEEGDataInfo.merge(InfoSub)
alliEEGDataInfo['Subject'] = alliEEGDataInfo['Subject'].str.rstrip('.mat')

hippocampal_contacts = alliEEGDataInfo.loc[alliEEGDataInfo['Label'].str.contains('Hippocampus')]
# hippocampal_contacts = hippocampal_contacts.query('Subject == "G06"')

for subject, contact_df in hippocampal_contacts.groupby('Subject'):
    print(subject)
    f = h5py.File(bvs_ieeg_data_dir+subject+'.mat', 'r')
    contact_df['data'] = contact_df.apply(read_contact, axis=1, h5_file=f)
    eeg_data = np.vstack(contact_df['data'].values)

    eeg = xr.DataArray(
      eeg_data,
      dims=("channel", "sample"),
      coords={
          # "event": 1, #get_ripplelogic assumes more than 1D sample (because of epochs)
          "channel": pd.MultiIndex.from_frame(contact_df[['Electrode', 'Site', 'MNI_1', 'MNI_2', 'MNI_3', 'Label', 'SamplingRate']]),
          # https://github.com/pennmem/cmlreaders/blob/1d8261632bd75ebd914d0c6e86cae1187b78b409/cmlreaders/eeg_container.py#L93
          "sample": np.arange(eeg_data.shape[1])
      },
    )
    
    choices = pd.read_csv(data_dir+subject+'_choices.csv')
    choice_events = pd.read_csv(data_dir+subject+'_choice_events.csv')
    choices = choices.reset_index(names='idTrial_inBehavior_withinTask')
    choices['idTrial_inBehavior_withinTask'] = choices['idTrial_inBehavior_withinTask'] + 1
    choices['Subject'] = subject
    # 81 = choice onset, 83 = choice
    choice_events_wide = choice_events.query('event_code in [81, 83]').pivot(
      index='idTrial_inBehavior_withinTask', columns='event_code', values='idx_in_iEEG')
    choice_events_wide.columns = ['idx_in_iEEG_' + str(col) for col in choice_events_wide.columns]
    choice_events_wide = choice_events_wide.reset_index()
    choice_events_wide['rel_samples'] = choice_events_wide['idx_in_iEEG_83'] - choice_events_wide['idx_in_iEEG_81']
    events = choices.merge(choice_events_wide)

    fixation_time = 3 #2.5 +- .5 
    if stim_locked:
        rel_start, rel_stop = 0-fixation_time, 10+fixation_time #in seconds
        offset_col = 'idx_in_iEEG_81'
    else:
        rel_start, rel_stop = -10-fixation_time, fixation_time #in seconds
        offset_col = 'idx_in_iEEG_83'
    sample_rate = contact_df['SamplingRate'].iloc[0]
    rel_start_samples = int(sample_rate * (rel_start))
    rel_stop_samples = int(sample_rate * (rel_stop))
    events['epoch_start'] = (events[offset_col] - 1) + rel_start_samples #epoch_start is presumably assuming MATLAB 1-indexed files
    events['epoch_stop'] = (events[offset_col] - 1) + rel_stop_samples

    time_frame = pd.DataFrame(
            {'rel_time': np.arange(rel_start_samples, rel_stop_samples) * (1 / sample_rate),
             'rel_sample': np.arange(rel_start_samples, rel_stop_samples)}
        ).reset_index(names='sample_in_epoch')

    indices = np.array(
        [eeg.sample.data[e[0] : (e[1] if e[1] > 0 else None)] for e in events[['epoch_start', 'epoch_stop']].values]
    )

    indexer = xr.DataArray(
        indices,
        dims=("event", "time"),
        coords={
            "event": pd.MultiIndex.from_frame(events), # Use the event index for the epoch coordinate
            "time": pd.MultiIndex.from_frame(time_frame)
        }
    )
    eeg_epoched = eeg.isel(sample=indexer)
    print(eeg_epoched)

    ripple_array = xr.apply_ufunc(ru.get_ripplelogic, 
                                  eeg_epoched, 
                                  kwargs={'sr': eeg_epoched.channel.SamplingRate.data[0],
                                          'times': eeg_epoched.time.rel_time.data},
                                  input_core_dims=[['event', 'time']],
                                  output_core_dims=[['event', 'time']],
                                  vectorize=True)
    print('ripples')
    start_array = xr.apply_ufunc(ru.getStartEndArrays, 
                                 ripple_array,
                                 kwargs={'start_only': True},
                                 input_core_dims=[['event', 'time']],
                                 output_core_dims=[['event', 'time']],
                                 vectorize=True)

    if not binned:
        total_ripple_df = start_array.to_dataframe(name='ripple').reset_index(drop=True)
        total_ripple_df.query('ripple != 0').to_csv(out_dir + subject + '_total_ripples.csv', index=False)

        deliberation_start_array = start_array.where((start_array.time['rel_sample'] >= 0) & 
          (start_array.time['rel_sample'] <= start_array.event['rel_samples']), 0)
        deliberation_total_ripples_array = deliberation_start_array.sum(dim='time')
        deliberation_total_df = deliberation_total_ripples_array.to_dataframe(name='ripples')
        deliberation_total_df.to_csv(out_dir + subject + '_deliberation_total_ripples.csv', index=False)

    elif stim_locked:
        resample_rate = 20
        times = np.round(np.arange(rel_start, rel_stop + (1 / resample_rate), (1 / resample_rate)), decimals=3)
        start_array_binned = start_array.drop_vars(['time', 'rel_sample', 'rel_time']).assign_coords(
          {'time': ripple_array.time.rel_time.data}
          ).groupby_bins('time', times, labels=times[:-1]).sum()
        start_array_df = start_array_binned.to_dataframe(name='ripple').reset_index(level=['time_bins']).reset_index(drop=True)
        start_array_df = start_array_df.query('time_bins >= -.5 and time_bins < (RT + .5)')
        start_array_df.to_csv(out_dir + subject + '_stim_locked_ripples.csv', index=False)
    else:
        resample_rate = 20
        times = np.round(np.arange(rel_start, rel_stop + (1 / resample_rate), (1 / resample_rate)), decimals=3)
        start_array_binned = start_array.drop_vars(['time', 'rel_sample', 'rel_time']).assign_coords(
          {'time': ripple_array.time.rel_time.data}
          ).groupby_bins('time', times, labels=times[:-1]).sum()
        start_array_df = start_array_binned.to_dataframe(name='ripple').reset_index(level=['time_bins']).reset_index(drop=True)
        start_array_df = start_array_df.query('time_bins >= -(RT + .5) and time_bins < .5')
        start_array_df.to_csv(out_dir + subject + '_response_locked_ripples.csv', index=False)