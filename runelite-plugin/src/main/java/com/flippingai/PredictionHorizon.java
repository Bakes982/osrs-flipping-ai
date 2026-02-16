package com.flippingai;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum PredictionHorizon
{
    ONE_MIN("1m", "1 Minute"),
    FIVE_MIN("5m", "5 Minutes"),
    THIRTY_MIN("30m", "30 Minutes"),
    TWO_HOUR("2h", "2 Hours"),
    EIGHT_HOUR("8h", "8 Hours"),
    TWENTY_FOUR_HOUR("24h", "24 Hours");

    private final String apiKey;
    private final String displayName;

    @Override
    public String toString()
    {
        return displayName;
    }
}
