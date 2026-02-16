package com.flippingai;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;
import net.runelite.client.config.ConfigSection;

@ConfigGroup("flippingai")
public interface FlippingAiConfig extends Config
{
    @ConfigSection(
        name = "Backend",
        description = "Backend connection settings",
        position = 0
    )
    String backendSection = "backend";

    @ConfigItem(
        keyName = "backendUrl",
        name = "Backend URL",
        description = "URL of the Flipping AI backend server",
        section = backendSection,
        position = 0
    )
    default String backendUrl()
    {
        return "http://localhost:8000";
    }

    @ConfigSection(
        name = "Display",
        description = "Overlay display settings",
        position = 1
    )
    String displaySection = "display";

    @ConfigItem(
        keyName = "showOverlay",
        name = "Show GE Overlay",
        description = "Show price suggestions on the GE interface",
        section = displaySection,
        position = 0
    )
    default boolean showOverlay()
    {
        return true;
    }

    @ConfigItem(
        keyName = "showConfidence",
        name = "Show Confidence",
        description = "Show ML confidence percentage on suggestions",
        section = displaySection,
        position = 1
    )
    default boolean showConfidence()
    {
        return true;
    }

    @ConfigItem(
        keyName = "horizon",
        name = "Prediction Horizon",
        description = "Which time horizon to display predictions for",
        section = displaySection,
        position = 2
    )
    default PredictionHorizon horizon()
    {
        return PredictionHorizon.FIVE_MIN;
    }

    @ConfigSection(
        name = "Notifications",
        description = "Alert settings",
        position = 2
    )
    String notifSection = "notifications";

    @ConfigItem(
        keyName = "notifyOnOpportunity",
        name = "Opportunity Alerts",
        description = "Show notification when a high-value flip opportunity is detected",
        section = notifSection,
        position = 0
    )
    default boolean notifyOnOpportunity()
    {
        return true;
    }

    @ConfigItem(
        keyName = "minProfitAlert",
        name = "Min Profit Alert (GP)",
        description = "Minimum expected profit to trigger an opportunity alert",
        section = notifSection,
        position = 1
    )
    default int minProfitAlert()
    {
        return 50000;
    }
}
