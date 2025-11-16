#!/bin/bash

apply_settings() {
    local json_file=$1
    local device=$2

    exp=$(jq '.exp' "$json_file")
    gain=$(jq '.gain' "$json_file")
    focus=$(jq '.focus' "$json_file")
    brightness=$(jq '.brightness' "$json_file")

    echo "Applying settings from $json_file to $device ..."

    v4l2-ctl -d "$device" -c exposure_time_absolute="$exp"
    v4l2-ctl -d "$device" -c gain="$gain"
    v4l2-ctl -d "$device" -c focus_absolute="$focus"
    v4l2-ctl -d "$device" -c brightness="$brightness"
}

apply_settings "../vision/config/camera_tune_results/best_camera_settings_kreo1.json" "/dev/video2"
apply_settings "../vision/config/camera_tune_results/best_camera_settings_kreo2.json" "/dev/video4"