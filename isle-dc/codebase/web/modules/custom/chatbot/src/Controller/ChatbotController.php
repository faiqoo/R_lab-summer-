<?php

namespace Drupal\chatbot\Controller;

use Drupal\Core\Controller\ControllerBase;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;
use GuzzleHttp\Exception\RequestException;

class ChatbotController extends ControllerBase {

  /**
   * Displays the chatbot page.
   */
  public function page() {
    return [
      '#theme' => 'chatbot_page',
      '#attached' => [
        'library' => [
          'chatbot/chatbot',
        ],
      ],
    ];
  }

  /**
   * Sends the user's message to the FastAPI backend.
   */
  public function sendMessage(Request $request) {

    $data = json_decode($request->getContent(), TRUE);

    $message = trim($data['message'] ?? '');

    if (empty($message)) {
      return new JsonResponse([
        'error' => 'Message cannot be empty.',
      ], 400);
    }

    try {

      $client = \Drupal::httpClient();

      $response = $client->post('http://host.docker.internal:8000/chat', [
        'headers' => [
          'Content-Type' => 'application/json',
          'X-Api-Key' => '4f5d40713235afe302297da1c9c310d5',
        ],
        'json' => [
          'question' => $message,
          'session_id' => 'test-session',
        ],
      ]);

      $body = json_decode($response->getBody()->getContents(), TRUE);

      return new JsonResponse([
        'reply' => $body['answer'] ?? 'No answer returned.',
      ]);

    }
    catch (RequestException $e) {

      $error = $e->getMessage();

      if ($e->hasResponse()) {
        $error = $e->getResponse()->getBody()->getContents();
      }

      \Drupal::logger('chatbot')->error($error);

      return new JsonResponse([
        'error' => 'Unable to contact chatbot backend.',
        'details' => $error,
      ], 500);
    }

    catch (\Exception $e) {

      \Drupal::logger('chatbot')->error($e->getMessage());

      return new JsonResponse([
        'error' => 'Unexpected error.',
        'details' => $e->getMessage(),
      ], 500);
    }

  }

}