#!/usr/bin/env bash
#

# User specific aliases and functions
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
#

#conda activate ${CONDA_ENVS_PATH}/$CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

source ./post_nwm_forcing_coastal.bash


post_nwm_forcing_coastal ${STARTPDY}${STARTCYC} \
	$DATAexec/coastal_forcing_output \
	$FCST_LENGTH_HRS \
        $NWM_FORCING_DIR




