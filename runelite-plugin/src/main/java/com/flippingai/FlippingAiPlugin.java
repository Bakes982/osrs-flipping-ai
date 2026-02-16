package com.flippingai;

import com.google.inject.Provides;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.runelite.api.Client;
import net.runelite.api.GrandExchangeOffer;
import net.runelite.api.GrandExchangeOfferState;
import net.runelite.api.events.GrandExchangeOfferChanged;
import net.runelite.api.events.ScriptPostFired;
import net.runelite.api.widgets.Widget;
import net.runelite.client.Notifier;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.events.ConfigChanged;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import net.runelite.client.ui.ClientToolbar;
import net.runelite.client.ui.NavigationButton;
import net.runelite.client.ui.overlay.OverlayManager;
import net.runelite.client.util.ImageUtil;

import javax.inject.Inject;
import java.awt.image.BufferedImage;
import java.util.concurrent.*;

@Slf4j
@PluginDescriptor(
    name = "Flipping AI",
    description = "AI-powered GE flip suggestions with multi-horizon price predictions",
    tags = {"grand exchange", "flipping", "money making", "ai", "predictions"}
)
public class FlippingAiPlugin extends Plugin
{
    @Inject
    private Client client;

    @Inject
    private ClientThread clientThread;

    @Inject
    private FlippingAiConfig config;

    @Inject
    private OverlayManager overlayManager;

    @Inject
    private ClientToolbar clientToolbar;

    @Inject
    private Notifier notifier;

    @Inject
    private FlippingAiOverlay overlay;

    @Getter
    private ApiClient apiClient;

    @Getter
    private ItemPrediction currentPrediction;

    @Getter
    private int currentGeItemId = -1;

    private FlippingAiPanel panel;
    private NavigationButton navButton;
    private ScheduledExecutorService scheduler;

    // GE widget group and child IDs
    private static final int GE_OFFER_WINDOW_GROUP = 465;
    private static final int GE_SEARCH_RESULTS = 162;
    private static final int GE_ITEM_ID_CHILD = 21;

    @Provides
    FlippingAiConfig provideConfig(ConfigManager configManager)
    {
        return configManager.getConfig(FlippingAiConfig.class);
    }

    @Override
    protected void startUp()
    {
        apiClient = new ApiClient(config.backendUrl());
        overlayManager.add(overlay);

        panel = new FlippingAiPanel(this);

        final BufferedImage icon = ImageUtil.loadImageResource(getClass(), "/panel_icon.png");
        navButton = NavigationButton.builder()
            .tooltip("Flipping AI")
            .icon(icon != null ? icon : new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB))
            .priority(10)
            .panel(panel)
            .build();
        clientToolbar.addNavigation(navButton);

        // Periodically refresh the panel opportunities
        scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(this::refreshPanel, 0, 30, TimeUnit.SECONDS);

        log.info("Flipping AI started - backend: {}", config.backendUrl());
    }

    @Override
    protected void shutDown()
    {
        overlayManager.remove(overlay);
        clientToolbar.removeNavigation(navButton);

        if (scheduler != null)
        {
            scheduler.shutdown();
        }
        if (apiClient != null)
        {
            apiClient.shutdown();
        }

        currentPrediction = null;
        currentGeItemId = -1;
        log.info("Flipping AI stopped");
    }

    @Subscribe
    public void onConfigChanged(ConfigChanged event)
    {
        if (!"flippingai".equals(event.getGroup()))
        {
            return;
        }
        if ("backendUrl".equals(event.getKey()))
        {
            apiClient.setBaseUrl(config.backendUrl());
        }
    }

    @Subscribe
    public void onScriptPostFired(ScriptPostFired event)
    {
        // Script 779 fires when GE offer setup changes (item selected)
        if (event.getScriptId() == 779)
        {
            updateGeItemId();
        }
    }

    @Subscribe
    public void onGrandExchangeOfferChanged(GrandExchangeOfferChanged event)
    {
        GrandExchangeOffer offer = event.getOffer();
        if (offer.getState() == GrandExchangeOfferState.BOUGHT ||
            offer.getState() == GrandExchangeOfferState.SOLD)
        {
            // Could log completed trades here for local tracking
            log.debug("GE trade completed: {} x{} @ {}gp",
                offer.getItemId(), offer.getTotalQuantity(), offer.getPrice());
        }
    }

    private void updateGeItemId()
    {
        clientThread.invokeLater(() -> {
            Widget geWidget = client.getWidget(GE_OFFER_WINDOW_GROUP, GE_ITEM_ID_CHILD);
            if (geWidget != null)
            {
                int itemId = geWidget.getItemId();
                if (itemId > 0 && itemId != currentGeItemId)
                {
                    currentGeItemId = itemId;
                    fetchPrediction(itemId);
                }
            }
        });
    }

    private void fetchPrediction(int itemId)
    {
        String horizon = config.horizon().getApiKey();
        apiClient.getPrediction(itemId, horizon).thenAccept(prediction -> {
            currentPrediction = prediction;
            if (prediction != null && config.notifyOnOpportunity() &&
                prediction.expectedProfit >= config.minProfitAlert())
            {
                notifier.notify("Flipping AI: " + prediction.itemName +
                    " - potential " + prediction.getFormattedProfit() + " GP profit!");
            }
        });
    }

    private void refreshPanel()
    {
        if (panel != null)
        {
            apiClient.getOpportunities(20).thenAccept(opps -> {
                if (opps != null)
                {
                    panel.updateOpportunities(opps);
                }
            });
        }
    }

    public void requestPrediction(int itemId)
    {
        fetchPrediction(itemId);
    }
}
