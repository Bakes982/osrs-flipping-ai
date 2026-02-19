package com.flippingai;

import com.google.inject.Provides;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.runelite.api.Client;
import net.runelite.api.GrandExchangeOffer;
import net.runelite.api.GrandExchangeOfferState;
import net.runelite.api.VarPlayer;
import net.runelite.api.events.GrandExchangeOfferChanged;
import net.runelite.api.events.ScriptPostFired;
import net.runelite.api.events.VarClientIntChanged;
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

    // GE price suggestion state
    private OfferEditorWidget offerEditor;
    private boolean isSelling = false;
    private boolean priceSuggestionsShown = false;

    // GE widget group and child IDs
    private static final int GE_OFFER_WINDOW_GROUP = 465;
    private static final int GE_SEARCH_RESULTS = 162;
    private static final int GE_ITEM_ID_CHILD = 21;
    private static final int GE_OFFER_CONTAINER_CHILD = 26;
    private static final int GE_OFFER_TYPE_TEXT_CHILD = 20;

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

        BufferedImage icon;
        try
        {
            icon = ImageUtil.loadImageResource(getClass(), "/panel_icon.png");
        }
        catch (Exception e)
        {
            log.warn("panel_icon.png not found, using fallback icon");
            icon = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
            // Draw a simple "F" as placeholder
            java.awt.Graphics2D g = icon.createGraphics();
            g.setColor(new java.awt.Color(0xF5, 0xA6, 0x23)); // amber
            g.fillRect(2, 2, 12, 12);
            g.setColor(java.awt.Color.BLACK);
            g.setFont(new java.awt.Font("Arial", java.awt.Font.BOLD, 11));
            g.drawString("F", 4, 13);
            g.dispose();
        }
        navButton = NavigationButton.builder()
            .tooltip("Flipping AI")
            .icon(icon)
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
        offerEditor = null;
        priceSuggestionsShown = false;
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
            log.debug("GE trade completed: {} x{} @ {}gp",
                offer.getItemId(), offer.getTotalQuantity(), offer.getPrice());
        }
    }

    @Subscribe
    public void onVarClientIntChanged(VarClientIntChanged event)
    {
        // VarClientInt index 5 = INPUT_TYPE
        if (event.getIndex() != 5)
        {
            return;
        }

        int inputType = client.getVarcIntValue(5);

        // Input closed — clean up
        if (inputType == 0)
        {
            offerEditor = null;
            priceSuggestionsShown = false;
            return;
        }

        // 7 = chatbox text input (used for GE price and quantity entry)
        if (inputType != 7 || !config.showPriceSuggestions())
        {
            return;
        }

        clientThread.invokeLater(this::tryCreatePriceSuggestions);
    }

    private void tryCreatePriceSuggestions()
    {
        if (priceSuggestionsShown)
        {
            return;
        }

        // Only show on the price input, not the quantity input
        Widget chatboxTitle = client.getWidget(162, 1);
        if (chatboxTitle == null || !"Set a price for each item:".equals(chatboxTitle.getText()))
        {
            return;
        }

        // Verify GE offer setup is visible
        Widget offerContainer = client.getWidget(GE_OFFER_WINDOW_GROUP, GE_OFFER_CONTAINER_CHILD);
        if (offerContainer == null || offerContainer.isHidden())
        {
            return;
        }

        // Determine buy/sell (varbit 4397 = GE_OFFER_CREATION_TYPE; 0=buy, 1=sell)
        isSelling = client.getVarbitValue(4397) == 1;

        // Get the current item from VarPlayer
        int geItemId = client.getVarpValue(VarPlayer.CURRENT_GE_ITEM);
        if (geItemId <= 0)
        {
            return;
        }

        // If we don't have a prediction for this item, fetch one and wait
        if (currentPrediction == null || currentPrediction.itemId != geItemId)
        {
            if (currentGeItemId != geItemId)
            {
                currentGeItemId = geItemId;
                fetchPrediction(geItemId);
            }
            return;
        }

        // Create the suggestion widgets in the chatbox
        Widget chatboxContainer = client.getWidget(162, 0);
        if (chatboxContainer == null)
        {
            return;
        }

        offerEditor = new OfferEditorWidget(client, chatboxContainer);
        // Show the opposite-side wiki price as the reference:
        // BUY  → insta-sell (floor / what sellers accept) matches Copilot UX
        // SELL → insta-buy  (ceiling / what buyers pay)
        long wikiPrice = isSelling ? currentPrediction.currentBuy : currentPrediction.currentSell;
        long aiPrice = isSelling ? currentPrediction.suggestedSell : currentPrediction.suggestedBuy;
        offerEditor.showPriceSuggestions(isSelling, wikiPrice, aiPrice);
        priceSuggestionsShown = true;
    }

    private void updateGeItemId()
    {
        clientThread.invokeLater(() -> {
            // Try VarPlayer first (more reliable), fall back to widget
            int itemId = client.getVarpValue(VarPlayer.CURRENT_GE_ITEM);
            if (itemId <= 0)
            {
                Widget geWidget = client.getWidget(GE_OFFER_WINDOW_GROUP, GE_ITEM_ID_CHILD);
                if (geWidget != null)
                {
                    itemId = geWidget.getItemId();
                }
            }
            if (itemId > 0 && itemId != currentGeItemId)
            {
                currentGeItemId = itemId;
                priceSuggestionsShown = false; // Reset when item changes
                fetchPrediction(itemId);
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
            // If the GE price input is still open, show suggestions now
            if (prediction != null)
            {
                clientThread.invokeLater(this::tryCreatePriceSuggestions);
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
